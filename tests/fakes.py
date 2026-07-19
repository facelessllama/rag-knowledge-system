"""
Lightweight fake stand-ins for api/main.py's heavy per-request services —
query_expander, retriever, reranker, generator. Paired with FastAPI's
app.dependency_overrides (see api/main.py's get_query_expander/get_retriever/
get_reranker/get_generator Depends() providers) to drive /query and
/query/stream through a real TestClient without ever constructing
EmbeddingService, CrossEncoderReranker, or an Ollama-backed LLMGenerator —
no GPU load, no model download, no network call.
"""


class FakeQueryExpander:
    async def expand(self, question: str) -> list[str]:
        return [question]

    async def contextualize(self, question: str, chat_history) -> str:
        return question


class FakeRetriever:
    def __init__(self, chunks: list[dict]):
        self._chunks = chunks

    async def retrieve_expanded(self, expanded_queries, top_k=20, folder=None, document_ids=None):
        return list(self._chunks)


class FakeReranker:
    """Mimics rag/reranker.py's CrossEncoderReranker.rerank() signature —
    called via run_on_gpu(reranker.rerank, query, chunks, top_k=...), so
    this must stay a plain sync callable, not async."""
    def rerank(self, query, chunks, top_k=3):
        ranked = sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)[:top_k]
        for c in ranked:
            c["rerank_score"] = c.get("score", 1.0)
        return ranked


class FakeGenerator:
    model = "fake-model"

    def __init__(self, answer: str = "This is a fake answer."):
        self._answer = answer

    async def generate_with_refusal_retry(self, messages, refusal_retries=2, model=None):
        return {
            "answer": self._answer,
            "model": model or self.model,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    async def generate_stream_with_refusal_retry(self, messages, refusal_retries=2, model=None):
        for word in self._answer.split():
            yield word + " "
