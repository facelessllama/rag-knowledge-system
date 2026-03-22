"""
Reranker Module
Re-scores retrieved chunks using a cross-encoder model.
"""
import logging
logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        logger.info(f"CrossEncoderReranker loaded: {model_name}")

    # If ALL cross-encoder scores are below this, the model can't handle the language
    # (ms-marco is English-only; Russian text scores cluster around -10 to -12)
    LANGUAGE_FALLBACK_THRESHOLD = -5.0

    def rerank(self, query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
        if not chunks:
            return []

        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)

        if max(scores) < self.LANGUAGE_FALLBACK_THRESHOLD:
            # Cross-encoder is blind to this language — keep vector score order
            logger.info(
                f"Reranker language fallback (max score={max(scores):.2f}) — "
                f"using vector scores for top {top_k}"
            )
            for chunk in chunks:
                chunk["rerank_score"] = chunk.get("score", 0)
        else:
            for chunk, score in zip(chunks, scores):
                chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        logger.info(
            f"Reranked {len(chunks)} → top {top_k} | "
            f"scores: {[round(c['rerank_score'], 3) for c in reranked[:top_k]]}"
        )
        return reranked[:top_k]


# Keep SimpleReranker as fallback
class SimpleReranker:
    def rerank(self, query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
        import re
        STOP_WORDS = {
            "a","an","the","and","or","but","in","on","at","to","for","of","with",
            "by","from","is","are","was","were","be","been","being","have","has",
            "had","do","does","did","will","would","could","should","may","might",
            "shall","can","this","that","these","those","it","its","as","if","not",
            "no","nor","so","yet","both","either","each","any","all","some",
        }
        def stem(t):
            for s in ("ment","tion","ing","ness","ies","ied","ed","er","ly","es","s"):
                if t.endswith(s) and len(t)-len(s) >= 4:
                    return t[:-len(s)]
            return t
        def tokens(text):
            return [stem(t) for t in re.sub(r"[^a-z0-9\s]"," ",text.lower()).split()
                    if len(t) > 1 and t not in STOP_WORDS]

        qt = tokens(query)
        qs = set(qt)
        qb = {(qt[i],qt[i+1]) for i in range(len(qt)-1)}

        for c in chunks:
            tt = tokens(c["text"])
            ts = set(tt)
            tb = {(tt[i],tt[i+1]) for i in range(len(tt)-1)}
            overlap = len(qs & ts) / len(qs) if qs else 0.0
            bigram  = len(qb & tb) / len(qb) if qb else 0.0
            phrase  = 0.15 if query.lower() in c["text"].lower() else 0.0
            hybrid  = 0.05 if c.get("source") == "hybrid" else 0.0
            penalty = 0.05 if c.get("source") == "neighbor" else 0.0
            base    = min(c.get("score", 0), 1.0)
            c["rerank_score"] = base*0.50 + overlap*0.28 + bigram*0.12 + phrase*0.10 + hybrid*0.05 - penalty

        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
