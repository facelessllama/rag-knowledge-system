"""
Hybrid Retriever
Combines vector search (semantic) + BM25 (keyword) for best results.
"""
import logging
from rank_bm25 import BM25Okapi
from embeddings.embedding_service import EmbeddingService
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
        self._bm25_index = None
        self._bm25_chunks = []
        logger.info(f"HybridRetriever ready | top_k={top_k}")

    def index_chunks_for_bm25(self, chunks: list):
        self._bm25_chunks = chunks
        tokenized = [chunk["text"].lower().split() for chunk in chunks]
        self._bm25_index = BM25Okapi(tokenized)
        logger.info(f"BM25 index built: {len(chunks)} chunks")

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """Single query retrieval"""
        k = top_k or self.top_k
        query_vector = self.embedder.embed_text(query)
        vector_results = self.vector_store.search(query_vector, top_k=k * 2)
        bm25_results = self._bm25_search(query, top_k=k * 2)
        merged = self._merge_results(vector_results, bm25_results, top_k=k)
        logger.info(f"Query: '{query[:50]}' | vector={len(vector_results)} bm25={len(bm25_results)} merged={len(merged)}")
        return merged

    async def retrieve_expanded(self, queries: list[str], top_k: int = None) -> list[dict]:
        """
        Multi-query retrieval — search for each query variant,
        merge all results, return top_k best unique chunks.
        """
        k = top_k or self.top_k
        all_chunks = {}

        for i, query in enumerate(queries):
            query_vector = self.embedder.embed_text(query)
            vector_results = self.vector_store.search(query_vector, top_k=k * 2)
            bm25_results = self._bm25_search(query, top_k=k * 2)

            # Скор дополнительных запросов чуть ниже
            weight = 1.0 if i == 0 else 0.85

            for r in vector_results + bm25_results:
                key = r.get("chunk_id", r["text"][:50])
                if key in all_chunks:
                    # Если чанк найден несколькими запросами — буст
                    all_chunks[key]["score"] += r["score"] * weight * 0.3
                else:
                    r = r.copy()
                    r["score"] = r["score"] * weight
                    all_chunks[key] = r

        # Сортируем и расширяем соседями
        merged = sorted(all_chunks.values(), key=lambda x: x["score"], reverse=True)
        expanded = self._expand_with_neighbors(merged)
        expanded.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"Expanded retrieval: {len(queries)} queries → {len(expanded)} chunks")
        return expanded[:k]

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        if not self._bm25_index:
            return []
        tokenized_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = self._bm25_chunks[idx].copy()
                chunk["score"] = float(scores[idx])
                chunk["source"] = "bm25"
                results.append(chunk)
        return results

    def _merge_results(self, vector_results: list, bm25_results: list, top_k: int) -> list[dict]:
        seen_chunks = {}
        for r in vector_results:
            key = r.get("chunk_id", r["text"][:50])
            r["source"] = "vector"
            seen_chunks[key] = r
        for r in bm25_results:
            key = r.get("chunk_id", r["text"][:50])
            if key in seen_chunks:
                seen_chunks[key]["score"] += r["score"] * 0.3
                seen_chunks[key]["source"] = "hybrid"
            else:
                seen_chunks[key] = r
        merged = sorted(seen_chunks.values(), key=lambda x: x["score"], reverse=True)
        expanded = self._expand_with_neighbors(merged)
        expanded.sort(key=lambda x: x["score"], reverse=True)
        return expanded[:top_k]

    def _expand_with_neighbors(self, chunks: list[dict]) -> list[dict]:
        expanded = {c.get("chunk_id", c["text"][:50]): c for c in chunks}
        for chunk in chunks:
            doc_id = chunk.get("document_id")
            chunk_index = chunk.get("chunk_index")
            if doc_id is None or chunk_index is None:
                continue
            for offset in [-1, 1]:
                neighbor_index = chunk_index + offset
                if neighbor_index < 0:
                    continue
                for c in self._bm25_chunks:
                    if (c.get("document_id") == doc_id and c.get("chunk_index") == neighbor_index):
                        key = c.get("chunk_id", c["text"][:50])
                        if key not in expanded:
                            neighbor = c.copy()
                            neighbor["score"] = chunk["score"] * 0.85
                            neighbor["source"] = "neighbor"
                            expanded[key] = neighbor
                        break
        return list(expanded.values())
