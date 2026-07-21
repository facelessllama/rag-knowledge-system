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


# EU-legislation CELEX IDs (see ingestion/chunker.py::extract_celex_id) —
# sector digit(1) + year(4) + type letter(1) + number(4), optional "(NN)"
# suffix. Applied to free query text, same role _CASE_NUMBER_RE plays for
# case numbers: a user typing "31997R0955" into a question needs this
# parsed out to drive the exact-identity lookup below.
_CELEX_ID_QUERY_RE = re.compile(r'\b(\d{5}[A-Z]\d{4}(?:\(\d+\))?)(?!\w)')


def extract_celex_ids(text: str) -> set[str]:
    return set(_CELEX_ID_QUERY_RE.findall(text))


# Natural short-citation form as a user actually types it in a query —
# "No 480/86", "No. 780/2007", "Regulation No 1021/2008" — distinct from
# the CELEX ID a real user is unlikely to know (see ingestion/chunker.py::
# extract_citation_number, which stores the ingestion-time counterpart of
# this on every chunk). Requires the slash so this can't collide with
# _CASE_NUMBER_RE's "<N> of <YYYY>" shape.
_CITATION_NUMBER_QUERY_RE = re.compile(r'\bNo\.?\s*(\d+)\s*/\s*(\d{2,4})\b')


def extract_citation_numbers(text: str) -> set[tuple[str, str]]:
    """Parses (number, year) short-citation pairs out of free query text —
    e.g. "Regulation (EC) No 480/86" -> {("480", "1986")}. 2-digit years
    are expanded the same way a reader conventionally resolves an EU
    short-citation (encountering "No 480/86" reads as 1986, not 2086),
    matching the full 4-digit year chunker.py::extract_citation_number
    stores at ingestion time, so both sides of the lookup agree."""
    out = set()
    for num, year in _CITATION_NUMBER_QUERY_RE.findall(text):
        if len(year) == 2:
            year = ("19" if int(year) >= 30 else "20") + year
        out.add((num, year))
    return out


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


# Structural Figure-N/Table-N references — see retrieve_expanded's use of
# extract_structural_references/best_structural_chunk. Mirrors the case-
# number/CELEX-ID/citation-number identity-lookup pattern above (parse an
# exact identifier out of the query, then find it verbatim rather than
# hoping embedding/BM25 similarity surfaces it), but for a signal that
# only exists as literal chunk TEXT, never structured ingestion-time
# payload metadata — captions aren't extracted at ingestion, so this has
# to text-scan a document's chunks at query time instead of doing an
# indexed payload lookup.
_STRUCTURAL_REF_QUERY_RE = re.compile(r'\b(Fig(?:ure)?|Table)\.?\s+(\d+[a-zA-Z]?)\b')


def extract_structural_references(text: str) -> set[tuple[str, str]]:
    """Parses (kind, num) pairs like ("Figure", "7") or ("Table", "12a")
    out of free query text. kind is normalized to "Figure"/"Table" — the
    query's own wording ("Fig.", "Fig", "Figure") doesn't need to match a
    chunk's, only what number/kind it refers to."""
    out = set()
    for kind_raw, num in _STRUCTURAL_REF_QUERY_RE.findall(text):
        kind = "Table" if kind_raw.lower().startswith("table") else "Figure"
        out.add((kind, num))
    return out


# Same three-tier idea as eval/mixed_corpus/build_golden_dataset.py's
# caption-extraction rewrite (colon/pipe strongest, period next, bare-no-
# punctuation weakest — some venues print "Fig. 1 Caption text..." with no
# punctuation at all before the description), independently reimplemented
# here rather than imported: that module is eval-only tooling, and this
# only needs to DETECT a caption-strength match in an already-indexed
# chunk, not extract/bound the caption's own text the way the golden
# builder does.
_STRUCTURAL_CAPTION_STRONG_RE = re.compile(r'\b(Fig(?:ure)?|Table)\.?\s+(\d+[a-zA-Z]?)\s*[:|]\s*')
_STRUCTURAL_CAPTION_MEDIUM_RE = re.compile(r'\b(Fig(?:ure)?|Table)\.?\s+(\d+[a-zA-Z]?)\s*\.\s+(?=[A-Z])')
_STRUCTURAL_CAPTION_WEAK_RE = re.compile(r'\b(Fig(?:ure)?|Table)\.?\s+(\d+[a-zA-Z]?)\s+(?=[A-Z])')

# A true caption describes what the figure/table shows — it never opens
# with a discourse connective continuing an unrelated paragraph (the
# "...as discussed in Figure 1. Finally, we turn to..." shape: the
# punctuation looks exactly like a caption opener, but "Finally," is
# mid-paragraph prose). Same list as the golden-builder rewrite, kept in
# sync deliberately — both exist to reject the same false-positive shape.
_STRUCTURAL_DISCOURSE_CONNECTIVES = (
    "finally", "however", "moreover", "therefore", "thus", "overall",
    "furthermore", "nevertheless", "meanwhile", "additionally",
    "nonetheless", "consequently", "hence", "similarly", "in addition",
    "in contrast", "on the other hand", "as a result",
)


def structural_match_tier(text: str, kind: str, num: str) -> int | None:
    """Best (lowest = strongest) tier at which `kind num` appears as a
    caption-like opener in this chunk's text, or None if it only appears
    as a plain in-text reference ("as shown in Figure 7...") or not at
    all."""
    for tier, pat in enumerate(
        (_STRUCTURAL_CAPTION_STRONG_RE, _STRUCTURAL_CAPTION_MEDIUM_RE, _STRUCTURAL_CAPTION_WEAK_RE)
    ):
        for m in pat.finditer(text):
            is_table = m.group(1).lower().startswith("table")
            if is_table != (kind == "Table") or m.group(2) != num:
                continue
            tail = text[m.end():m.end() + 30].lstrip().lower()
            if any(tail.startswith(w) for w in _STRUCTURAL_DISCOURSE_CONNECTIVES):
                continue
            return tier
    return None


def best_structural_chunk(chunks: list[dict], kind: str, num: str) -> dict | None:
    """The single best (lowest-tier) chunk among `chunks` (one document's
    full chunk list) that contains a caption-strength match for
    `kind num`, or None. Deliberately returns at most one — see
    retrieve_expanded's docstring on why only one structural chunk per
    document is ever added, guaranteed, rather than every match."""
    best_chunk, best_tier = None, None
    for chunk in chunks:
        tier = structural_match_tier(chunk.get("text", ""), kind, num)
        if tier is not None and (best_tier is None or tier < best_tier):
            best_tier, best_chunk = tier, chunk
    return best_chunk


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


def promote_missing_compare_documents(
    chunks: list[dict], top_chunks: list[dict], document_ids: list[str] | None
) -> list[dict]:
    """retrieve_expanded() guarantees every requested document_id reaches the
    candidate pool (see its own "at least 1 chunk per document" step), but
    the cross-encoder reranker scores each candidate independently against
    the (often long, multi-document) compare question — if one document's
    content scores lower across the board than another's, top_k reranking
    can cut it completely, even though it was present pre-rerank. Observed
    directly: comparing 2 documents, the reranker's top-20 was 100% one
    document, and the LLM correctly (but unhelpfully) reported the other
    "not found" despite it being retrievable and in scope. Silently
    comparing N-1 of N requested documents is worse than including one
    extra chunk per missing one — pull each missing document's single
    best-scoring candidate back in, using the rerank_score
    CrossEncoderReranker.rerank() already set on every chunk in `chunks` as
    a side effect (it scores the full input list before slicing to top_k)."""
    if not document_ids or len(document_ids) < 2:
        return top_chunks

    present = {c.get("document_id") for c in top_chunks}
    missing = [d for d in document_ids if d not in present]
    if not missing:
        return top_chunks

    promoted = list(top_chunks)
    for doc_id in missing:
        candidates = [c for c in chunks if c.get("document_id") == doc_id]
        if not candidates:
            continue
        best = max(candidates, key=lambda c: c.get("rerank_score", c.get("score", 0)))
        promoted.append(best)
    return promoted


class HybridRetriever:
    # Documents at or under this many total chunks get pulled in whole once
    # any one chunk of theirs scores well enough to be a top candidate — see
    # _expand_with_neighbors.
    SHORT_DOC_CHUNK_LIMIT = 10

    # How many of stage 1's top-scoring candidate documents the structural
    # Figure-N/Table-N lookup text-scans (see retrieve_expanded) — bounded
    # deliberately small: a query naming one figure/table almost always
    # means one specific document, and this is a per-document full-chunk
    # scroll + regex scan, not free.
    STRUCTURAL_TOP_DOCS = 3

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

    async def retrieve_expanded(self, queries: list[str], top_k: int = None, folder: str = None,
                                 document_ids: list[str] | set[str] | None = None) -> list[dict]:
        """
        Multi-query retrieval — search for each query variant,
        merge all results, return top_k best unique chunks.

        document_ids, when given, restricts retrieval to exactly that set —
        the multi-doc "compare" flow used to only ever *hint* at this via
        filenames spelled out in the question text plus a folder filter,
        which meant hybrid search was still free to surface (and the
        reranker free to keep) chunks from *other* documents in the same
        folder. Passed straight through to hybrid_search's doc_filter
        (Qdrant MatchAny) so out-of-scope chunks are never fetched in the
        first place, and re-applied as a filter on the identity-index hits
        below (case/CELEX/citation lookups don't take a doc filter
        themselves) so no path around this method can reintroduce them.
        """
        k = top_k or self.top_k
        # Bring a large pool of candidates to the reranker
        pool = max(30, k * 5)
        all_chunks = {}
        doc_scope = set(document_ids) if document_ids else None
        # queries[0] is always the original (unexpanded) user question — see
        # query_expander.expand().
        query_case_numbers = extract_case_numbers(queries[0]) if queries else set()
        query_celex_ids = extract_celex_ids(queries[0]) if queries else set()
        query_citation_numbers = extract_citation_numbers(queries[0]) if queries else set()
        query_structural_refs = extract_structural_references(queries[0]) if queries else set()

        for i, query in enumerate(queries):
            query_vector = await run_on_gpu(self.embedder.embed_text, query)
            results = await asyncio.to_thread(
                self.vector_store.hybrid_search, query_vector, query, top_k=pool,
                doc_filter=list(doc_scope) if doc_scope else None, folder_filter=folder or None
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

        # Same guarantee for CELEX IDs (see vector_db/qdrant_client.py::
        # chunks_by_celex_id) — more load-bearing here than case numbers
        # are for the court-case corpus, since a CELEX ID never appears in
        # its own document's body text at all (EU legislation cites itself
        # in a different format), so hybrid search has no literal string to
        # find without this index.
        for celex_id in query_celex_ids:
            indexed = await asyncio.to_thread(self.vector_store.chunks_by_celex_id, celex_id)
            for r in indexed:
                key = r.get("chunk_id", r["text"][:50])
                if key not in all_chunks:
                    r = r.copy()
                    r["score"] = 1.0
                    r["source"] = "celex_id_index"
                    all_chunks[key] = r

        # Same guarantee for the natural short-citation form ("No 480/86")
        # — see ingestion/chunker.py::extract_citation_number and
        # eval/README.md, "ID-free spot-check @ 57,000": this is the path
        # a real user actually relies on (unlike the CELEX ID above, which
        # they're unlikely to know), and it was the one measurably weak
        # spot found once CELEX-ID-shortcut queries were excluded from
        # testing — the short number alone collides across different
        # years and gets cited in passing by *other* documents, so hybrid
        # search alone isn't reliable for it without this index.
        for num, year in query_citation_numbers:
            indexed = await asyncio.to_thread(self.vector_store.chunks_by_citation_number, num, year)
            for r in indexed:
                key = r.get("chunk_id", r["text"][:50])
                if key not in all_chunks:
                    r = r.copy()
                    r["score"] = 1.0
                    r["source"] = "citation_number_index"
                    all_chunks[key] = r

        # Structural Figure-N/Table-N lookup — only when the query itself
        # names one explicitly ("What does Figure 7 show...").
        # Unlike the case/CELEX/citation-number indexes above, a caption
        # isn't structured ingestion-time metadata — chunks_for_document()
        # pulls a candidate document's full chunk list and best_structural_
        # chunk() text-scans it for the strongest-looking caption match,
        # adding AT MOST ONE chunk per document (never a wider pool — a
        # caption is either in one specific chunk of a document or it
        # isn't, unlike the reverted two-stage retrieval attempt this
        # deliberately doesn't repeat; see project memory on why blindly
        # widening the candidate pool made evidence-page recall worse, not
        # better).
        #
        # Which documents get scanned differs by whether the caller scoped
        # the query:
        # - doc_scope set (the "user already has a document open"/compare
        #   flow) — scan EVERY explicitly selected document, not just a
        #   stage-1-inferred top few. These documents are already user-
        #   confirmed relevant, not a guess from embedding/BM25 score, so
        #   the ambiguity the top-N heuristic below exists to bound
        #   doesn't apply. Measured gap this closes: a title-free,
        #   document-scoped caption question (the realistic "open a
        #   document, ask about one of its figures" product workflow) used
        #   to skip this lookup entirely — Evidence-chunk Recall on that
        #   exact scenario was 69.5% vs. 91.5% unscoped-with-title, purely
        #   because of the skip (see project memory / eval/mixed_corpus/
        #   README.md's title-free robustness check). Bounded by
        #   MAX_DOCUMENT_IDS (see api/schemas.py, 50) rather than
        #   STRUCTURAL_TOP_DOCS — the caller already committed to this
        #   many documents by asking for them.
        # - doc_scope unset (a normal open-ended question) — scan only the
        #   top STRUCTURAL_TOP_DOCS candidate documents by stage-1 score,
        #   same as before this change.
        if query_structural_refs:
            if doc_scope:
                candidate_doc_ids = list(doc_scope)
            else:
                doc_scores: dict[str, float] = {}
                for c in all_chunks.values():
                    doc_id = c.get("document_id")
                    if doc_id:
                        doc_scores[doc_id] = max(doc_scores.get(doc_id, 0.0), c["score"])
                candidate_doc_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)[:self.STRUCTURAL_TOP_DOCS]

            for doc_id in candidate_doc_ids:
                doc_chunks = await asyncio.to_thread(self.vector_store.chunks_for_document, doc_id)
                best_chunk = None
                for kind, num in query_structural_refs:
                    candidate = best_structural_chunk(doc_chunks, kind, num)
                    if candidate is not None:
                        best_chunk = candidate
                        break  # a query names one Figure/Table almost always — first match is enough
                if best_chunk is not None:
                    # Promote the canonical instance of this chunk regardless
                    # of whether hybrid search already found it — NOT "inject
                    # if absent". A structural match is a stronger signal
                    # than any embedding/BM25 score; that has to hold whether
                    # hybrid search missed the chunk entirely OR already
                    # surfaced it with a weak score. The old "if key not in
                    # all_chunks" guard only injected when absent, so a
                    # chunk hybrid search had already found (however weakly)
                    # never got upgraded — earlier, weaker detection silently
                    # defeated the stronger structural signal instead of
                    # deferring to it (confirmed on 112/127 real caption
                    # misses — see project memory's taxonomy pass). Matches
                    # the case/CELEX/citation-number identity-index
                    # convention above: identity_match=True (below)
                    # guarantees this chunk survives retrieve_expanded's own
                    # top_k slice even if the reranker would've scored it
                    # low — extended here to a signal that only exists as
                    # chunk text, not a stored payload field.
                    key = best_chunk.get("chunk_id", best_chunk["text"][:50])
                    matched = all_chunks.get(key)
                    if matched is None:
                        matched = best_chunk.copy()
                        all_chunks[key] = matched
                    matched["score"] = 1.0
                    matched["source"] = "structural_reference"
                    matched["identity_match"] = True

        # The case/CELEX/citation-number index lookups above don't take a
        # doc filter themselves (they're exact-key lookups, not searches) —
        # enforce document_ids here too so a compare scoped to specific
        # documents can't have an out-of-scope one slip back in via an
        # identity match (e.g. two of the picked documents share a citation
        # number with a third, unselected one).
        if doc_scope:
            all_chunks = {k: v for k, v in all_chunks.items() if v.get("document_id") in doc_scope}

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
            elif c.get("celex_id") and c["celex_id"] in query_celex_ids:
                c["identity_match"] = True
            elif (c.get("citation_number"), c.get("citation_year")) in query_citation_numbers:
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
