"""
Tests for rag/generator.py's GeneratorRouter — the single place that
decides whether a request's data ever reaches the DeepSeek cloud
generator. See its own docstring for the two independent gates
(administrator cloud_enabled + per-request provider="deepseek") this
suite exists to prove, both individually and never bypassed.

tests/test_query_endpoint.py's "cloud-mode opt-in gating" section proves
the same guarantees end-to-end through the real HTTP endpoint; these are
the corresponding unit-level tests of the router itself.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag.generator import BaseGenerator, DeepSeekGenerator, GeneratorRouter, ProviderNotAvailable


class _FakeBackend(BaseGenerator):
    model = "fake-model"

    def __init__(self, answer="answer"):
        self._answer = answer
        self.generate = AsyncMock(return_value={
            "answer": answer, "model": self.model,
            "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
        })


def _stream_backend(tokens=("a", "b")):
    backend = _FakeBackend()

    async def _stream(messages, refusal_retries=2, model=None):
        for t in tokens:
            yield t
    backend.generate_stream_with_refusal_retry = _stream
    return backend


# ── generate_with_refusal_retry: gating ──────────────────────────────────────

@pytest.mark.asyncio
async def test_default_provider_none_uses_local():
    local, cloud = _FakeBackend("local"), _FakeBackend("cloud")
    router = GeneratorRouter(local=local, deepseek=cloud, cloud_enabled=True)
    result = await router.generate_with_refusal_retry([{"role": "user", "content": "q"}])
    assert result["answer"] == "local"
    assert result["provider"] == "local"
    cloud.generate.assert_not_called()


@pytest.mark.asyncio
async def test_explicit_provider_local_uses_local_even_with_cloud_enabled():
    local, cloud = _FakeBackend("local"), _FakeBackend("cloud")
    router = GeneratorRouter(local=local, deepseek=cloud, cloud_enabled=True)
    result = await router.generate_with_refusal_retry([{"role": "user", "content": "q"}], provider="local")
    assert result["provider"] == "local"
    cloud.generate.assert_not_called()


@pytest.mark.asyncio
async def test_explicit_provider_deepseek_reaches_cloud_when_fully_enabled():
    local, cloud = _FakeBackend("local"), _FakeBackend("cloud")
    router = GeneratorRouter(local=local, deepseek=cloud, cloud_enabled=True)
    result = await router.generate_with_refusal_retry([{"role": "user", "content": "q"}], provider="deepseek")
    assert result["answer"] == "cloud"
    assert result["provider"] == "deepseek"
    local.generate.assert_not_called()


@pytest.mark.asyncio
async def test_deepseek_ignores_client_supplied_model():
    """Regression test: a request can send provider="deepseek" alongside
    `model` set to whatever Ollama model the UI's local model picker had
    selected (e.g. "qwen2.5:7b") — the UI itself no longer sends this
    combination (see frontend/app.js's runQueryNonStreaming), but the
    server must not depend on that. The router must drop the client model
    for any non-local backend so the administrator-configured DeepSeek
    model is always the one actually used."""
    local, cloud = _FakeBackend("local"), _FakeBackend("cloud")
    router = GeneratorRouter(local=local, deepseek=cloud, cloud_enabled=True)
    await router.generate_with_refusal_retry(
        [{"role": "user", "content": "q"}], model="qwen2.5:7b", provider="deepseek"
    )
    cloud.generate.assert_awaited_once_with([{"role": "user", "content": "q"}], model=None)
    local.generate.assert_not_called()


@pytest.mark.asyncio
async def test_provider_deepseek_raises_when_admin_has_not_enabled_cloud():
    """A configured deepseek backend alone is not enough — the
    administrator gate (cloud_enabled) must ALSO be on."""
    local, cloud = _FakeBackend("local"), _FakeBackend("cloud")
    router = GeneratorRouter(local=local, deepseek=cloud, cloud_enabled=False)
    with pytest.raises(ProviderNotAvailable):
        await router.generate_with_refusal_retry([{"role": "user", "content": "q"}], provider="deepseek")
    local.generate.assert_not_called()
    cloud.generate.assert_not_called()


@pytest.mark.asyncio
async def test_provider_deepseek_raises_when_no_backend_configured():
    """cloud_enabled=True but no API key was ever configured (deepseek is
    None) — same clear error, never a silent switch to local."""
    local = _FakeBackend("local")
    router = GeneratorRouter(local=local, deepseek=None, cloud_enabled=True)
    with pytest.raises(ProviderNotAvailable):
        await router.generate_with_refusal_retry([{"role": "user", "content": "q"}], provider="deepseek")
    local.generate.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_provider_name_raises():
    local = _FakeBackend("local")
    router = GeneratorRouter(local=local, deepseek=None, cloud_enabled=False)
    with pytest.raises(ProviderNotAvailable):
        await router.generate_with_refusal_retry([{"role": "user", "content": "q"}], provider="openai")
    local.generate.assert_not_called()


def test_model_property_reflects_local_backend_regardless_of_cloud_state():
    """api/main.py reads router.model directly for the two early-return
    responses that never invoke a generator at all (no chunks retrieved /
    below relevance threshold) — must always be the local model, since no
    provider was ever actually chosen in that path."""
    local, cloud = _FakeBackend("local"), _FakeBackend("cloud")
    cloud.model = "cloud-model"
    router = GeneratorRouter(local=local, deepseek=cloud, cloud_enabled=True)
    assert router.model == "fake-model"


def test_ollama_url_property_delegates_to_local():
    """Regression test: api/main.py's _check_ollama() health check reads
    generator.ollama_url directly — GeneratorRouter used to have no such
    attribute at all (an AttributeError caught by _check_ollama's bare
    `except Exception`, silently reporting Ollama as down even though it
    was healthy) until this property was added."""
    local = _FakeBackend("local")
    local.ollama_url = "http://localhost:11435"
    router = GeneratorRouter(local=local, deepseek=None, cloud_enabled=False)
    assert router.ollama_url == "http://localhost:11435"


# ── generate_stream_with_refusal_retry: local-only ───────────────────────────

@pytest.mark.asyncio
async def test_stream_default_provider_streams_from_local():
    local = _stream_backend(("h", "i"))
    router = GeneratorRouter(local=local, deepseek=None, cloud_enabled=False)
    tokens = [t async for t in router.generate_stream_with_refusal_retry([{"role": "user", "content": "q"}])]
    assert tokens == ["h", "i"]


@pytest.mark.asyncio
async def test_stream_provider_deepseek_raises_even_when_fully_enabled():
    """Streaming is local-only regardless of cloud_enabled/backend state —
    DeepSeekGenerator has no generate_stream at all, and this must be a
    clear error, not an AttributeError deep inside the router or a silent
    switch to local."""
    local, cloud = _stream_backend(), _FakeBackend("cloud")
    router = GeneratorRouter(local=local, deepseek=cloud, cloud_enabled=True)
    with pytest.raises(ProviderNotAvailable):
        async for _ in router.generate_stream_with_refusal_retry(
            [{"role": "user", "content": "q"}], provider="deepseek"
        ):
            pass


# ── BaseGenerator.generate_with_refusal_retry is shared by both backends ────

@pytest.mark.asyncio
async def test_deepseek_generator_gets_refusal_retry_for_free():
    """DeepSeekGenerator used to have NO refusal-retry at all (it was
    eval-only tooling before becoming a real, user-selectable provider) —
    now inherits the same policy LLMGenerator already had, via
    BaseGenerator, so cloud-mode answers get the same treatment local ones
    do."""
    from rag.generator import DeepSeekGenerator

    gen = DeepSeekGenerator(api_key="fake-key")
    gen.generate = AsyncMock(side_effect=[
        {"answer": "I could not find this information in the provided documents.",
         "model": "deepseek-v4-flash", "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        {"answer": "The answer is 42.", "model": "deepseek-v4-flash",
         "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    ])
    result = await gen.generate_with_refusal_retry([{"role": "user", "content": "q"}], refusal_retries=2)
    assert result["answer"] == "The answer is 42."
    assert gen.generate.await_count == 2


@pytest.mark.asyncio
async def test_deepseek_sends_thinking_enabled_in_payload():
    """Regression test: the actual HTTP payload sent to DeepSeek must ask
    for thinking="enabled", matching the reasoning mode the DeepSeek A/B
    benchmark (eval/mixed_corpus/README.md) was actually measured under —
    that benchmark never set this field, so it ran against the provider
    default, which DeepSeek's own API docs state is "enabled". Sending
    "disabled" here would run production under an unbenchmarked mode."""
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={
        "choices": [{"message": {"content": "42"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })
    client = AsyncMock()
    client.post = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    gen = DeepSeekGenerator(api_key="fake-key")
    with patch("httpx.AsyncClient", return_value=client):
        await gen.generate([{"role": "user", "content": "q"}])

    sent_json = client.post.call_args.kwargs["json"]
    assert sent_json["thinking"] == {"type": "enabled"}
