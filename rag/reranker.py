"""
Reranker Module
Re-scores retrieved chunks for better relevance ordering.
"""
import logging
logger = logging.getLogger(__name__)


class SimpleReranker:
    def rerank(self, query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
        """Rerank chunks by relevance to query"""
        query_terms = set(query.lower().split())

        for chunk in chunks:
            text_lower = chunk["text"].lower()
            text_terms = set(text_lower.split())

            # Процент слов запроса найденных в чанке
            term_overlap = len(query_terms & text_terms) / max(len(query_terms), 1)

            # Бонус если чанк найден и вектором и BM25
            hybrid_bonus = 0.15 if chunk.get("source") == "hybrid" else 0.0

            # Бонус за точное вхождение фразы запроса
            phrase_bonus = 0.2 if query.lower() in text_lower else 0.0

            chunk["rerank_score"] = (
                chunk.get("score", 0) * 0.6 +
                term_overlap * 0.25 +
                phrase_bonus * 0.1 +
                hybrid_bonus * 0.05
            )

        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        logger.info(f"Reranked {len(chunks)} chunks → top {top_k}")
        return reranked[:top_k]
