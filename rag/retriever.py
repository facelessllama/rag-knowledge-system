"""
Hybrid Retriever
Combines vector search (semantic) + BM25 (keyword) for best results.
"""
import logging
import re
from rank_bm25 import BM25Okapi
from embeddings.embedding_service import EmbeddingService
from vector_db.qdrant_client import VectorStore

logger = logging.getLogger(__name__)

_CYRILLIC_RE = re.compile(r"[а-яё]")

# Words that carry no search signal
STOP_WORDS_EN = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "as", "if", "not", "no", "nor",
    "so", "yet", "both", "either", "each", "any", "all", "some",
}

STOP_WORDS_RU = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а",
    "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же",
    "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от",
    "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже",
    "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был",
    "него", "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там",
    "потом", "себя", "ничего", "ей", "может", "они", "тут", "где", "есть",
    "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам",
    "чтоб", "без", "будто", "чего", "раз", "тоже", "себе", "под",
    "будет", "ж", "тогда", "кто", "этот", "того", "потому", "этого",
    "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой",
    "тем", "чтобы", "нее", "были", "куда", "зачем", "всех", "никогда",
    "можно", "при", "наконец", "два", "об", "другой", "хоть", "после",
    "над", "больше", "тот", "через", "эти", "нас", "про", "всего",
    "него", "какая", "много", "разве", "три", "эту", "моя", "впрочем",
    "хорошо", "свою", "этой", "перед", "иногда", "лучше", "чуть",
    "том", "нельзя", "такой", "им", "более", "всегда", "конечно",
    "всю", "между",
}

STOP_WORDS = STOP_WORDS_EN | STOP_WORDS_RU


def _stem(token: str) -> str:
    for suffix in ("ment", "tion", "ing", "ness", "ies", "ied", "ed", "er", "ly", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop words. Stems English tokens only."""
    lowered = text.lower()
    tokens = re.sub(r"[^a-zа-яё0-9\s]", " ", lowered).split()
    result = []
    for t in tokens:
        if len(t) <= 1 or t in STOP_WORDS:
            continue
        # Stem only ASCII (English) tokens; Russian morphology is too complex for a suffix stripper
        result.append(t if _CYRILLIC_RE.search(t) else _stem(t))
    return result


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
        tokenized = [_tokenize(chunk["text"]) for chunk in chunks]
        self._bm25_index = BM25Okapi(tokenized)
        logger.info(f"BM25 index built: {len(chunks)} chunks")

    def retrieve(self, query: str, top_k: int = None, folder: str = None) -> list[dict]:
        k = top_k or self.top_k
        pool = max(30, k * 5)
        query_vector = self.embedder.embed_text(query)
        vector_results = self.vector_store.search(query_vector, top_k=pool, folder_filter=folder or None)
        bm25_results = self._bm25_search(query, top_k=pool, folder=folder)
        merged = self._merge_results(vector_results, bm25_results, top_k=k)
        logger.info(f"Query: '{query[:50]}' | vector={len(vector_results)} bm25={len(bm25_results)} merged={len(merged)}")
        return merged

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
            query_vector = self.embedder.embed_text(query)
            vector_results = self.vector_store.search(query_vector, top_k=pool, folder_filter=folder or None)
            bm25_results = self._bm25_search(query, top_k=pool, folder=folder)

            # Slightly down-weight secondary query variants
            weight = 1.0 if i == 0 else 0.9

            for r in vector_results + bm25_results:
                key = r.get("chunk_id", r["text"][:50])
                if key in all_chunks:
                    # Chunk found by multiple queries — boost proportionally
                    all_chunks[key]["score"] = min(
                        1.0, all_chunks[key]["score"] + r["score"] * weight * 0.25
                    )
                    if r.get("source") == "vector" and all_chunks[key].get("source") == "bm25":
                        all_chunks[key]["source"] = "hybrid"
                    elif r.get("source") == "bm25" and all_chunks[key].get("source") == "vector":
                        all_chunks[key]["source"] = "hybrid"
                else:
                    r = r.copy()
                    r["score"] = r["score"] * weight
                    all_chunks[key] = r

        # Sort and expand with neighbors only for top candidates
        sorted_chunks = sorted(all_chunks.values(), key=lambda x: x["score"], reverse=True)
        expanded = self._expand_with_neighbors(sorted_chunks, max_base=pool)
        expanded.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"Expanded retrieval: {len(queries)} queries → {len(expanded)} candidates → top {k}")
        return expanded[:k]

    def _bm25_search(self, query: str, top_k: int, folder: str = None) -> list[dict]:
        if not self._bm25_index:
            return []
        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []
        scores = self._bm25_index.get_scores(tokenized_query)

        # Filter candidate indices by folder BEFORE sorting
        if folder:
            candidate_indices = [
                i for i, c in enumerate(self._bm25_chunks)
                if c.get("folder", "") == folder
            ]
        else:
            candidate_indices = range(len(scores))

        # Normalize BM25 scores to [0, 1] range over the candidate set
        candidate_scores = [scores[i] for i in candidate_indices]
        max_score = max(candidate_scores) if candidate_scores else 0
        if max_score <= 0:
            return []

        top_indices = sorted(candidate_indices, key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = self._bm25_chunks[idx].copy()
                chunk["score"] = float(scores[idx]) / max_score  # normalized [0, 1]
                chunk["source"] = "bm25"
                results.append(chunk)
        return results

    def _merge_results(self, vector_results: list, bm25_results: list, top_k: int) -> list[dict]:
        seen_chunks = {}
        for r in vector_results:
            key = r.get("chunk_id", r["text"][:50])
            r = r.copy()
            r["source"] = "vector"
            seen_chunks[key] = r
        for r in bm25_results:
            key = r.get("chunk_id", r["text"][:50])
            if key in seen_chunks:
                # Average the two normalized scores + hybrid bonus
                seen_chunks[key]["score"] = (seen_chunks[key]["score"] + r["score"]) / 2 + 0.05
                seen_chunks[key]["source"] = "hybrid"
            else:
                seen_chunks[key] = r.copy()

        merged = sorted(seen_chunks.values(), key=lambda x: x["score"], reverse=True)
        expanded = self._expand_with_neighbors(merged, max_base=top_k * 3)
        expanded.sort(key=lambda x: x["score"], reverse=True)
        return expanded[:top_k]

    def _expand_with_neighbors(self, chunks: list[dict], max_base: int = 20) -> list[dict]:
        """Add adjacent chunks for top candidates only to avoid noise."""
        expanded = {c.get("chunk_id", c["text"][:50]): c for c in chunks}

        # Only expand neighbors for the top-scoring candidates
        top_candidates = chunks[:max_base]
        for chunk in top_candidates:
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
                            # Neighbors score lower than direct hits
                            neighbor["score"] = chunk["score"] * 0.6
                            neighbor["source"] = "neighbor"
                            expanded[key] = neighbor
                        break
        return list(expanded.values())
