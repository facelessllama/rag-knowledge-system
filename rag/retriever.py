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

from embeddings.embedding_service import EmbeddingService
from rag.executors import run_on_gpu
from vector_db.qdrant_client import VectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
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

        # Sort and expand with neighbors only for top candidates
        sorted_chunks = sorted(all_chunks.values(), key=lambda x: x["score"], reverse=True)
        expanded = await self._expand_with_neighbors(sorted_chunks, max_base=pool)
        expanded.sort(key=lambda x: x["score"], reverse=True)

        # Ensure at least 1 chunk per unique document in the result
        top_k_chunks = expanded[:k]
        docs_in_top = {c.get("document_id") for c in top_k_chunks}
        for c in expanded[k:]:
            doc_id = c.get("document_id")
            if doc_id and doc_id not in docs_in_top:
                top_k_chunks.append(c)
                docs_in_top.add(doc_id)

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
