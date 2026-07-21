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
    """A fake single-backend generator (stands in for either LLMGenerator
    or DeepSeekGenerator) — wrap one or two of these in a REAL
    rag.generator.GeneratorRouter in tests rather than re-implementing the
    provider-gating logic here, so endpoint tests exercise the actual
    gating code (see tests/test_generator_router.py for direct unit tests
    of the router itself). call_count lets a test assert a specific
    backend was (or, more importantly, was NOT) invoked."""
    model = "fake-model"

    def __init__(self, answer: str = "This is a fake answer."):
        self._answer = answer
        self.call_count = 0

    async def generate(self, messages, retries=3, model=None):
        self.call_count += 1
        return {
            "answer": self._answer,
            "model": model or self.model,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    async def generate_with_refusal_retry(self, messages, refusal_retries=2, model=None):
        return await self.generate(messages, model=model)

    async def generate_stream_with_refusal_retry(self, messages, refusal_retries=2, model=None):
        self.call_count += 1
        for word in self._answer.split():
            yield word + " "
