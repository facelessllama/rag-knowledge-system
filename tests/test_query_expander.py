"""
Tests for rag/query_expander.py's contextualize() — rewrites a follow-up
question into a standalone one using chat history, so retrieval (which
never sees chat_history itself, only this method's return value) can
resolve references like "the court" or "this order" to what they actually
mean. Added after observing live that a follow-up's best_rerank_score was
identical (-1.55, vs. a 3.0 threshold) whether or not chat_history was even
included in the request — proof retrieval was ignoring it entirely.

The underlying Ollama call is mocked (httpx.AsyncClient) — no live model
needed. expand()'s own tests aren't duplicated here; this file is only
about the new method.
"""
from unittest.mock import AsyncMock, MagicMock, patch

from rag.query_expander import QueryExpander


def _mock_ollama_response(content: str):
    """Builds a fake httpx.AsyncClient context manager whose .post() returns
    an Ollama-shaped chat response with the given assistant content."""
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={"message": {"content": content}})

    client = AsyncMock()
    client.post = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


async def test_contextualize_is_a_noop_with_no_history():
    """The common case (first message, no follow-up) must cost zero LLM
    calls — not just return the query unchanged, but never touch the
    network at all."""
    qe = QueryExpander()
    with patch("httpx.AsyncClient") as mock_cls:
        result = await qe.contextualize("What is the case about?", None)
        mock_cls.assert_not_called()
    assert result == "What is the case about?"


async def test_contextualize_is_a_noop_with_empty_history_list():
    qe = QueryExpander()
    with patch("httpx.AsyncClient") as mock_cls:
        result = await qe.contextualize("What is the case about?", [])
        mock_cls.assert_not_called()
    assert result == "What is the case about?"


async def test_contextualize_resolves_a_pronoun_using_history():
    history = [
        {"role": "user", "content": "What is the case Sabir Ansari vs The State of Bihar about?"},
        {"role": "assistant", "content": "The case involves Sabir Ansari, a petitioner against the State of Bihar..."},
    ]
    qe = QueryExpander()
    with patch("httpx.AsyncClient", return_value=_mock_ollama_response(
        "What was the court's final decision in the Sabir Ansari vs The State of Bihar case?"
    )):
        result = await qe.contextualize("What was the court's final decision?", history)

    assert "Sabir Ansari" in result
    assert result != "What was the court's final decision?"


async def test_contextualize_passes_through_an_already_standalone_question():
    history = [
        {"role": "user", "content": "What is the case Sabir Ansari vs The State of Bihar about?"},
        {"role": "assistant", "content": "..."},
    ]
    qe = QueryExpander()
    with patch("httpx.AsyncClient", return_value=_mock_ollama_response(
        "When does the Traffic Management Order 2012 come into force?"
    )):
        result = await qe.contextualize("When does the Traffic Management Order 2012 come into force?", history)

    assert result == "When does the Traffic Management Order 2012 come into force?"


async def test_contextualize_falls_back_to_original_on_llm_failure():
    """A worse retrieval query beats a broken request — same defensive
    posture as expand()'s own except-and-fall-back."""
    history = [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    qe = QueryExpander()
    with patch("httpx.AsyncClient", side_effect=RuntimeError("Ollama unreachable")):
        result = await qe.contextualize("What was the court's final decision?", history)

    assert result == "What was the court's final decision?"


async def test_contextualize_falls_back_when_response_is_blank():
    history = [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    qe = QueryExpander()
    with patch("httpx.AsyncClient", return_value=_mock_ollama_response("   ")):
        result = await qe.contextualize("What was the court's final decision?", history)

    assert result == "What was the court's final decision?"


async def test_contextualize_strips_surrounding_quotes_from_llm_output():
    history = [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    qe = QueryExpander()
    with patch("httpx.AsyncClient", return_value=_mock_ollama_response(
        '"What was the court\'s final decision in the Sabir Ansari case?"'
    )):
        result = await qe.contextualize("What was the court's final decision?", history)

    assert not result.startswith('"')
    assert not result.endswith('"')


async def test_contextualize_only_uses_the_last_two_exchanges():
    """Keeps the prompt small/fast — only the last 4 turns (2 exchanges) go
    into the contextualization prompt, not the whole conversation."""
    history = [{"role": "user", "content": f"turn {i}"} for i in range(10)]
    qe = QueryExpander()
    mock_client = _mock_ollama_response("standalone question")
    with patch("httpx.AsyncClient", return_value=mock_client):
        await qe.contextualize("follow-up", history)

    sent_prompt = mock_client.post.call_args.kwargs["json"]["messages"][0]["content"]
    assert "turn 9" in sent_prompt
    assert "turn 6" in sent_prompt  # last 4 of 10 turns: 6,7,8,9
    assert "turn 5" not in sent_prompt
