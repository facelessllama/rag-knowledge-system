"""
Hybrid Retriever
Dense (semantic) + sparse (BM25-style) search, fused server-side by Qdrant
(RRF over prefetch candidates). No client-side keyword index — see
vector_db/qdrant_client.py and vector_db/sparse_encoder.py.

All embedding (GPU) and Qdrant (I/O) calls run off the event loop thread
(see rag/executors.py) so concurrent requests don't serialize behind them.
"""
import asyncio
import logging
import re

from embeddings.embedding_service import EmbeddingService
from rag.executors import run_on_gpu
from vector_db.qdrant_client import VectorStore

logger = logging.getLogger(__name__)

# Case numbers in this corpus follow a near-universal "<N> of <YYYY>"
# convention (CRM 10691 of 2020, WPA 12091 of 2019, F.A.O. No.293 of 2019,
# ABLAPL 1380 of 2020, ...) regardless of the type-prefix abbreviation,
# which varies too much (dots, spaces, "No.") to normalize reliably. An
# exact (number, year) match is a much stronger identity signal than
# embedding/BM25 similarity for this corpus's short, heavily-templated
# interlocutory orders — many differ from each other in almost nothing
# else but this number (item date, "Heard learned counsel for...",
# signature block are near-identical across cases) — see retrieve_expanded.
_CASE_NUMBER_RE = re.compile(r'\b(\d{1,6})\s+of\s+(\d{4})\b')


def extract_case_numbers(text: str) -> set[tuple[str, str]]:
    """Parses (number, year) pairs out of free text — used on the *query*,
    which has no fixed structure to key off. For chunks, prefer the
    structured case_number/case_year payload fields (see
    ingestion/chunker.py::extract_case_metadata) instead of re-running this
    against chunk text — a chunk deep in a ruling's body won't always repeat
    the caption's case number even though it's unambiguously part of that
    case."""
    return set(_CASE_NUMBER_RE.findall(text))


def extract_party_match(query: str, parties: list[str] | None) -> bool:
    """True if every party name from a chunk's structured case metadata
    (ingestion/chunker.py::extract_case_metadata) appears in the query text.
    case_summary questions are literally built from these two names ("What
    is the case X vs Y about?"), so this is as strong an identity signal as
    an exact case-number match — but only when *all* parties match, not
    just one: many filenames in this corpus share a common party (e.g. "...
    vs STATE OF ODISHA"), so a single-name match would over-promote every
    case involving that party instead of the one actually asked about."""
    if not parties:
        return False
    query_lower = query.lower()
    return all(p.lower() in query_lower for p in parties)


def promote_identity_matches(
    chunks: list[dict], top_chunks: list[dict], relevance_threshold: float
) -> list[dict]:
    """retrieve_expanded() guarantees a chunk with an exact case-number or
    full-party-name match survives into the pool handed to the reranker —
    but the cross-encoder scores it independently and can still rank it low
    (these short, near-duplicate interlocutory orders are exactly where the
    reranker is least reliable), which would then have the caller's
    relevance-threshold gate reject it anyway. An exact identity match is a
    far more reliable signal than the cross-encoder's opinion for this
    corpus, so force it in with a score that clears the gate rather than
    trust reranking here."""
    matches = [c for c in chunks if c.get("identity_match")]
    if not matches:
        return top_chunks

    # top_chunks is already top_k-sized from reranking; only append matches
    # not already present (matches are rare — usually 0 or 1 per query) so
    # the result stays close to top_k without needing to re-sort-and-slice,
    # which would risk cutting a just-boosted match back out if other
    # chunks scored even higher (sorting by score before slicing defeats
    # the guarantee this function exists to make).
    promoted = list(top_chunks)
    included_ids = {c.get("chunk_id") for c in promoted}
    floor_score = relevance_threshold + 1.0

    for c in matches:
        if c.get("chunk_id") in included_ids:
            for tc in promoted:
                if tc.get("chunk_id") == c.get("chunk_id"):
                    tc["rerank_score"] = max(tc.get("rerank_score", 0), floor_score)
            continue
        c = c.copy()
        c["rerank_score"] = floor_score
        promoted.append(c)
        included_ids.add(c.get("chunk_id"))

    return promoted


def promote_document_opening_chunks(chunks: list[dict], top_chunks: list[dict]) -> list[dict]:
    """A document's opening chunk (chunk_index 0) usually carries its title/
    citation sentence — e.g. "This Order may be cited as the Localism Act
    2011 (Commencement No. 6 ...) and shall come into force on...". A later
    chunk that continues that same sentence ("...ame day on which it is
    made") reads as a fragment without it, even though the reranker often
    scores the opening chunk lower than surrounding content-specific chunks
    (it's short and formulaic) and cuts it from the final top_k. Root-caused
    directly: the identical retrieved+reranked chunk set, missing only this
    opening chunk, made a 7B model refuse to answer a plainly-answerable
    question in the majority of trials — see eval/README.md, "Known issue:
    multi-doc near-duplicate-title flakiness". If a document is already
    represented in top_chunks and its opening chunk exists somewhere in the
    broader candidate pool, include it too — cheap, and it's exactly the
    "citation context" a reader would otherwise get for free from a
    document's own layout."""
    doc_ids_in_top = {c.get("document_id") for c in top_chunks}
    included_keys = {c.get("chunk_id", c["text"][:50]) for c in top_chunks}

    opening_chunks = {}
    for c in chunks:
        if c.get("document_id") in doc_ids_in_top and c.get("chunk_index") == 0:
            opening_chunks.setdefault(c["document_id"], c)

    promoted = list(top_chunks)
    for doc_id, opening in opening_chunks.items():
        key = opening.get("chunk_id", opening["text"][:50])
        if key not in included_keys:
            promoted.append(opening.copy())
            included_keys.add(key)

    return promoted


class HybridRetriever:
    # Documents at or under this many total chunks get pulled in whole once
    # any one chunk of theirs scores well enough to be a top candidate — see
    # _expand_with_neighbors.
    SHORT_DOC_CHUNK_LIMIT = 10

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        top_k: int = 5
    ):
        self.embedder = embedding_service
        self.vector_store = vector_store
        self.top_k = top_k
        logger.info(f"HybridRetriever ready | top_k={top_k}")

    async def retrieve(self, query: str, top_k: int = None, folder: str = None) -> list[dict]:
        k = top_k or self.top_k
        pool = max(30, k * 5)
        query_vector = await run_on_gpu(self.embedder.embed_text, query)
        results = await asyncio.to_thread(
            self.vector_store.hybrid_search, query_vector, query, top_k=pool, folder_filter=folder or None
        )
        for r in results:
            r["source"] = "hybrid"
        expanded = await self._expand_with_neighbors(results, max_base=pool)
        expanded.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"Query: '{query[:50]}' | hybrid={len(results)} expanded={len(expanded)}")
        return expanded[:k]

    async def retrieve_expanded(self, queries: list[str], top_k: int = None, folder: str = None) -> list[dict]:
        """
        Multi-query retrieval — search for each query variant,
        merge all results, return top_k best unique chunks.
        """
        k = top_k or self.top_k
        # Bring a large pool of candidates to the reranker
        pool = max(30, k * 5)
        all_chunks = {}
        # queries[0] is always the original (unexpanded) user question — see
        # query_expander.expand().
        query_case_numbers = extract_case_numbers(queries[0]) if queries else set()

        for i, query in enumerate(queries):
            query_vector = await run_on_gpu(self.embedder.embed_text, query)
            results = await asyncio.to_thread(
                self.vector_store.hybrid_search, query_vector, query, top_k=pool, folder_filter=folder or None
            )

            # Slightly down-weight secondary query variants
            weight = 1.0 if i == 0 else 0.9

            for r in results:
                key = r.get("chunk_id", r["text"][:50])
                if key in all_chunks:
                    # Chunk found by multiple queries — boost proportionally
                    all_chunks[key]["score"] = min(
                        1.0, all_chunks[key]["score"] + r["score"] * weight * 0.25
                    )
                else:
                    r = r.copy()
                    r["score"] = r["score"] * weight
                    r["source"] = "hybrid"
                    all_chunks[key] = r

        # An exact (case number, year) match is a far stronger identity
        # signal than any embedding/BM25 score can provide — fetch its
        # chunks directly via Qdrant's case_number/case_year payload index
        # (see vector_db/qdrant_client.py::chunks_by_case_number) instead of
        # hoping hybrid search's top-`pool` candidates happen to include
        # them, a bet that gets worse as the corpus grows and more documents
        # compete for the same pool slots. Scored at the cap so these
        # survive both the neighbor-expansion cutoff below and the final
        # top_k slice.
        for num, year in query_case_numbers:
            indexed = await asyncio.to_thread(self.vector_store.chunks_by_case_number, num, year)
            for r in indexed:
                key = r.get("chunk_id", r["text"][:50])
                if key not in all_chunks:
                    r = r.copy()
                    r["score"] = 1.0
                    r["source"] = "case_number_index"
                    all_chunks[key] = r

        # Sort and expand with neighbors only for top candidates
        sorted_chunks = sorted(all_chunks.values(), key=lambda x: x["score"], reverse=True)
        expanded = await self._expand_with_neighbors(sorted_chunks, max_base=pool)

        # Mark exact-identity matches from each chunk's own structured case
        # metadata (extracted from its filename at ingestion — see
        # ingestion/chunker.py::extract_case_metadata), not by regexing the
        # chunk's own text: the number/parties are a property of the
        # *document*, and a chunk deep in a ruling's body won't always
        # repeat them even though it's unambiguously part of that case.
        query_text = queries[0] if queries else ""
        for c in expanded:
            if c.get("case_number") and (c["case_number"], c.get("case_year")) in query_case_numbers:
                c["identity_match"] = True
            elif extract_party_match(query_text, c.get("parties")):
                c["identity_match"] = True

        expanded.sort(key=lambda x: x["score"], reverse=True)

        # Ensure at least 1 chunk per unique document in the result
        top_k_chunks = expanded[:k]
        docs_in_top = {c.get("document_id") for c in top_k_chunks}
        for c in expanded[k:]:
            doc_id = c.get("document_id")
            if doc_id and doc_id not in docs_in_top:
                top_k_chunks.append(c)
                docs_in_top.add(doc_id)

        # An exact case-number or full-party-name match is a far stronger
        # identity signal than any embedding/BM25 score — guarantee it
        # survives even if its semantic score alone wouldn't have made the
        # cut (these short, near-duplicate interlocutory orders are exactly
        # where semantic scoring is least reliable).
        included_keys = {c.get("chunk_id", c["text"][:50]) for c in top_k_chunks}
        for c in expanded:
            key = c.get("chunk_id", c["text"][:50])
            if c.get("identity_match") and key not in included_keys:
                top_k_chunks.append(c)
                included_keys.add(key)

        logger.info(f"Expanded retrieval: {len(queries)} queries → {len(expanded)} candidates → top {len(top_k_chunks)}")
        return top_k_chunks

    async def _expand_with_neighbors(self, chunks: list[dict], max_base: int = 20) -> list[dict]:
        """Add adjacent chunks for top candidates only, fetched from Qdrant directly
        (bounded per-document lookup — not a scan over the whole corpus)."""
        expanded = {c.get("chunk_id", c["text"][:50]): c for c in chunks}

        # Only expand neighbors for the top-scoring candidates
        top_candidates = chunks[:max_base]

        by_doc: dict[str, dict[int, dict]] = {}
        for chunk in top_candidates:
            doc_id = chunk.get("document_id")
            chunk_index = chunk.get("chunk_index")
            if doc_id is None or chunk_index is None:
                continue
            by_doc.setdefault(doc_id, {})[chunk_index] = chunk

        for doc_id, index_map in by_doc.items():
            # Short documents (e.g. a 2-page bail order — caption + parties on
            # one chunk, the actual ruling on another): a ±1 index window can
            # miss the substantive chunk entirely if only the caption chunk
            # scored well on a generic query, leaving the LLM with just names
            # and a case number. Pull in the whole document instead when it's
            # cheap to (<= SHORT_DOC_CHUNK_LIMIT chunks total) rather than
            # gambling on index adjacency.
            full_doc = await asyncio.to_thread(
                self.vector_store.all_chunks_for_document, doc_id, self.SHORT_DOC_CHUNK_LIMIT
            )
            if full_doc is not None:
                best_score = max(c["score"] for c in index_map.values())
                for chunk in full_doc:
                    key = chunk.get("chunk_id", chunk["text"][:50])
                    if key in expanded:
                        continue
                    chunk = chunk.copy()
                    chunk["score"] = best_score * 0.6
                    chunk["source"] = "neighbor"
                    expanded[key] = chunk
                continue

            wanted_indices = set()
            for chunk_index in index_map:
                for offset in (-1, 1):
                    neighbor_index = chunk_index + offset
                    if neighbor_index >= 0:
                        wanted_indices.add(neighbor_index)
            if not wanted_indices:
                continue

            neighbors = await asyncio.to_thread(self.vector_store.neighbor_chunks, doc_id, sorted(wanted_indices))
            for neighbor in neighbors:
                key = neighbor.get("chunk_id", neighbor["text"][:50])
                if key in expanded:
                    continue
                # Attribute the neighbor's score to whichever adjacent top candidate scored higher
                anchors = [
                    index_map.get(neighbor["chunk_index"] - 1),
                    index_map.get(neighbor["chunk_index"] + 1),
                ]
                anchors = [a for a in anchors if a is not None]
                if not anchors:
                    continue
                base_chunk = max(anchors, key=lambda c: c["score"])
                neighbor = neighbor.copy()
                # Neighbors score lower than direct hits
                neighbor["score"] = base_chunk["score"] * 0.6
                neighbor["source"] = "neighbor"
                expanded[key] = neighbor

        return list(expanded.values())
