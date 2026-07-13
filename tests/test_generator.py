"""
Tests for rag/generator.py — leaked <question> tag stripping, refusal
detection, and refusal-retry (resample-on-refusal for both the
non-streaming and streaming generate paths — see LLMGenerator's
generate_with_refusal_retry/generate_stream_with_refusal_retry docstrings
and eval/README.md's "Known issue: multi-doc near-duplicate-title
flakiness" for why this exists).
"""
from unittest.mock import AsyncMock

import pytest

from rag.generator import LLMGenerator, is_refusal, strip_leaked_question_tag


def test_strips_leaked_question_tag_prefix():
    text = "<question>Where is it banned?</question> It is banned in several regions."
    assert strip_leaked_question_tag(text) == "It is banned in several regions."


def test_leaves_normal_answer_untouched():
    text = "It is banned in several regions."
    assert strip_leaked_question_tag(text) == text


def test_only_strips_leading_occurrence():
    text = "<question>Q</question> Answer mentions <question>nested</question> literally."
    result = strip_leaked_question_tag(text)
    assert result == "Answer mentions <question>nested</question> literally."


def test_multiline_question_tag_content():
    text = "<question>Line one\nLine two?</question>\nActual answer here."
    assert strip_leaked_question_tag(text) == "Actual answer here."


# ── is_refusal ─────────────────────────────────────────────────────────────────

def test_is_refusal_matches_known_openers():
    assert is_refusal("I could not find this information in the provided documents.")
    assert is_refusal("I couldn't find relevant information in the knowledge base.")
    assert is_refusal("No relevant information found in the knowledge base.")


def test_is_refusal_is_case_insensitive_and_strips_whitespace():
    assert is_refusal("  i COULD not FIND anything useful.")


def test_is_refusal_false_for_a_real_answer():
    assert not is_refusal("The order comes into force on 1st April 2013.")


# ── generate_with_refusal_retry ─────────────────────────────────────────────────

def _result(answer):
    return {"answer": answer, "model": "m", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


@pytest.mark.asyncio
async def test_refusal_retry_returns_immediately_when_not_a_refusal():
    gen = LLMGenerator()
    gen.generate = AsyncMock(return_value=_result("The date is 1st April 2013."))

    result = await gen.generate_with_refusal_retry([{"role": "user", "content": "q"}])

    assert result["answer"] == "The date is 1st April 2013."
    gen.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_refusal_retry_resamples_until_a_real_answer():
    gen = LLMGenerator()
    gen.generate = AsyncMock(side_effect=[
        _result("I could not find this information in the provided documents."),
        _result("I could not find this information in the provided documents."),
        _result("The date is 1st April 2013."),
    ])

    result = await gen.generate_with_refusal_retry([{"role": "user", "content": "q"}], refusal_retries=2)

    assert result["answer"] == "The date is 1st April 2013."
    assert gen.generate.await_count == 3


@pytest.mark.asyncio
async def test_refusal_retry_gives_up_after_max_retries():
    gen = LLMGenerator()
    gen.generate = AsyncMock(return_value=_result("I could not find this information in the provided documents."))

    result = await gen.generate_with_refusal_retry([{"role": "user", "content": "q"}], refusal_retries=2)

    assert is_refusal(result["answer"])
    assert gen.generate.await_count == 3  # 1 initial + 2 retries


# ── generate_stream_with_refusal_retry ──────────────────────────────────────────

def _token_streams(*streams):
    """Returns a generate_stream() replacement that yields each `streams`
    list, one per call, in order."""
    iterator = iter(streams)

    async def fake_generate_stream(messages, model=None):
        for token in next(iterator):
            yield token

    return fake_generate_stream


async def _collect(async_gen):
    return "".join([t async for t in async_gen])


@pytest.mark.asyncio
async def test_stream_refusal_retry_passes_through_normal_answer_unbuffered():
    gen = LLMGenerator()
    gen.generate_stream = _token_streams(["The date ", "is 1st ", "April 2013."])

    result = await _collect(gen.generate_stream_with_refusal_retry([{"role": "user", "content": "q"}]))

    assert result == "The date is 1st April 2013."


@pytest.mark.asyncio
async def test_stream_refusal_retry_discards_refusal_and_streams_retry_success():
    gen = LLMGenerator()
    gen.generate_stream = _token_streams(
        ["I could not find ", "this information in the provided documents."],
        ["The date is 1st ", "April 2013."],
    )

    result = await _collect(gen.generate_stream_with_refusal_retry(
        [{"role": "user", "content": "q"}], refusal_retries=2))

    assert result == "The date is 1st April 2013."


@pytest.mark.asyncio
async def test_stream_refusal_retry_streams_refusal_after_exhausting_retries():
    gen = LLMGenerator()
    refusal = "I could not find this information in the provided documents."
    gen.generate_stream = _token_streams([refusal], [refusal], [refusal])

    result = await _collect(gen.generate_stream_with_refusal_retry(
        [{"role": "user", "content": "q"}], refusal_retries=2))

    assert result == refusal


@pytest.mark.asyncio
async def test_stream_refusal_retry_does_not_drop_a_short_non_refusal_stream():
    """A response that ends mid-buffer, still a strict prefix of a known
    refusal opener ('I could not' is a prefix of 'I could not find') and
    therefore never resolved True/False inside the loop, must fall back to
    a definitive check and be flushed — not silently dropped."""
    gen = LLMGenerator()
    gen.generate_stream = _token_streams(["I could not"])

    result = await _collect(gen.generate_stream_with_refusal_retry([{"role": "user", "content": "q"}]))

    assert result == "I could not"
