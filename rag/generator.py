"""
Handles communication with LLM backend.

Uses Ollama's OpenAI-compatible API endpoint
so switching to real OpenAI requires only URL change.
"""
import logging
import re
import httpx
import asyncio

logger = logging.getLogger(__name__)

# Some models (observed on qwen3) occasionally echo the <question> wrapper
# used to isolate user input (see prompt_builder.py) instead of just
# answering — strip it defensively even though the system prompt now also
# instructs against it.
_LEAKED_QUESTION_TAG = re.compile(r'^\s*<question>.*?</question>\s*', re.DOTALL)


def strip_leaked_question_tag(text: str) -> str:
    return _LEAKED_QUESTION_TAG.sub('', text, count=1)


# Canned refusal openers used both by the model (system prompt instructs it
# to start an unanswerable response this way) and by api/main.py's own
# below-threshold short-circuit message — see is_refusal().
REFUSAL_PREFIXES = ("i could not find", "i couldn't find", "no relevant information")
_MAX_REFUSAL_PREFIX_LEN = max(len(p) for p in REFUSAL_PREFIXES)


def is_refusal(answer: str) -> bool:
    return answer.strip().lower().startswith(REFUSAL_PREFIXES)


class PartialStreamError(Exception):
    """Raised when the LLM stream breaks after chunks have already been sent to the client."""
    def __init__(self, chunks_yielded: int):
        super().__init__(f"Stream interrupted after partial output ({chunks_yielded} chunks)")
        self.chunks_yielded = chunks_yielded


class DeepSeekGenerator:
    """Same generate() interface as LLMGenerator (drop-in for accuracy A/B
    testing against the local Ollama model — see eval/test_deepseek_accuracy.py)
    but talks to DeepSeek's real OpenAI-compatible endpoint (Bearer auth,
    top-level temperature/max_tokens, response in choices[0].message.content)
    rather than Ollama's native /api/chat shape (think/options wrapper),
    which LLMGenerator actually uses despite its module docstring's claim of
    OpenAI-compatibility. Not wired into api/main.py's default generator —
    intentionally a separate, explicitly-opted-into path for now."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.1,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        logger.info(f"DeepSeekGenerator ready | model={model}")

    async def generate(self, messages: list[dict], retries: int = 3, model: str = None) -> dict:
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=5.0)
                ) as client:
                    active_model = model or self.model
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": active_model,
                            "messages": messages,
                            "stream": False,
                            "temperature": self.temperature,
                            "max_tokens": 1024,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    usage = data.get("usage", {})
                    return {
                        "answer": strip_leaked_question_tag(data["choices"][0]["message"]["content"]),
                        "model": active_model,
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"DeepSeek timeout attempt {attempt}/{retries}")
                if attempt < retries:
                    await asyncio.sleep(2 * attempt)
            except httpx.ConnectError as e:
                last_error = e
                logger.warning(f"DeepSeek connect error attempt {attempt}/{retries}")
                if attempt < retries:
                    await asyncio.sleep(2 * attempt)
            except httpx.HTTPStatusError as e:
                logger.error(f"DeepSeek API error {e.response.status_code}: {e.response.text[:300]}")
                raise
        logger.error(f"DeepSeek failed after {retries} attempts")
        raise last_error


class LLMGenerator:
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        temperature: float = 0.1
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.temperature = temperature
        logger.info(f"LLMGenerator ready | model={model} temp={temperature}")

    async def generate(self, messages: list[dict], retries: int = 3, model: str = None) -> dict:
        """
        Send messages to LLM and get response.
        Retries up to 3 times on timeout or connection error.
        """
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        connect=10.0,
                        read=120.0,
                        write=10.0,
                        pool=5.0
                    )
                ) as client:
                    active_model = model or self.model
                    response = await client.post(
                        f"{self.ollama_url}/api/chat",
                        json={
                            "model": active_model,
                            "messages": messages,
                            "stream": False,
                            # Reasoning models (qwen3, deepseek-r1, ...) emit a hidden
                            # <think> pass by default — 5-10x more tokens, no benefit
                            # for RAG Q&A since retrieval already did the "thinking".
                            # Ignored by non-reasoning models (qwen2.5, ...).
                            "think": False,
                            "options": {
                                "temperature": self.temperature,
                                "num_predict": 1024
                            }
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    if attempt > 1:
                        logger.info(f"LLM succeeded on attempt {attempt}")
                    return {
                        "answer": strip_leaked_question_tag(data["message"]["content"]),
                        "model": active_model,
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                        "total_tokens": (
                            data.get("prompt_eval_count", 0) +
                            data.get("eval_count", 0)
                        )
                    }
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"LLM timeout attempt {attempt}/{retries}")
                if attempt < retries:
                    await asyncio.sleep(2 * attempt)
            except httpx.ConnectError as e:
                last_error = e
                logger.warning(f"LLM connect error attempt {attempt}/{retries}")
                if attempt < retries:
                    await asyncio.sleep(2 * attempt)
            except Exception as e:
                logger.error(f"LLM unexpected error: {e}")
                raise
        logger.error(f"LLM failed after {retries} attempts")
        raise httpx.TimeoutException(
            f"LLM did not respond after {retries} attempts. Is Ollama running?"
        )

    async def generate_with_refusal_retry(
        self, messages: list[dict], refusal_retries: int = 2, model: str = None
    ) -> dict:
        """Callers only reach this after their own relevance-threshold gate
        already passed (see api/main.py) — i.e. retrieval was confident
        enough to attempt an answer. A refusal at that point is usually
        wrong, not conservative: measured directly (eval/README.md, "Known
        issue: multi-doc near-duplicate-title flakiness") — the identical
        prompt, resampled, answers correctly a clear majority of the time
        for cases that clear the gate. Root cause wasn't pinned down to a
        deterministic prompt/ordering issue (it didn't reproduce reliably
        in isolated scripted tests, unlike the A.B.A. 493 case fixed
        earlier), so this is a pragmatic mitigation for what looks like
        inference-level nondeterminism rather than a prompt rewrite chasing
        a moving target. Cheap: only fires on the rare refusal, and only
        resamples the same already-built prompt."""
        result = await self.generate(messages, model=model)
        attempt = 0
        while is_refusal(result["answer"]) and attempt < refusal_retries:
            attempt += 1
            logger.info(f"Refusal despite confident retrieval — resampling ({attempt}/{refusal_retries})")
            result = await self.generate(messages, model=model)
        return result

    async def generate_stream(self, messages: list[dict], model: str = None, retries: int = 3):
        """
        Stream response token by token.
        Retries up to 3 times on timeout or connection error.
        Raises on final failure so caller can send a proper error event.
        """
        import json
        last_error = None
        for attempt in range(1, retries + 1):
            chunks_yielded = 0
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=5.0)) as client:
                    async with client.stream(
                        "POST",
                        f"{self.ollama_url}/api/chat",
                        json={
                            "model": model or self.model,
                            "messages": messages,
                            "stream": True,
                            "think": False,
                            "options": {"temperature": self.temperature}
                        }
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON from Ollama: {line!r}")
                                continue
                            if data.get("done"):
                                break
                            content = data.get("message", {}).get("content")
                            if content:
                                chunks_yielded += 1
                                yield content
                return  # success
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                if chunks_yielded > 0:
                    logger.error(f"LLM stream interrupted after {chunks_yielded} chunks, not retrying")
                    raise PartialStreamError(chunks_yielded) from e
                logger.warning(f"LLM stream {type(e).__name__} attempt {attempt}/{retries} (no chunks sent)")
                if attempt < retries:
                    await asyncio.sleep(2 * attempt)
            except Exception as e:
                logger.error(f"LLM stream error: {e}")
                raise
        logger.error(f"LLM stream failed after {retries} attempts")
        raise last_error

    async def generate_stream_with_refusal_retry(
        self, messages: list[dict], refusal_retries: int = 2, model: str = None
    ):
        """Streaming counterpart to generate_with_refusal_retry() — see its
        docstring for why a refusal is worth resampling here. True token
        streaming can't be "retried" once tokens have reached the client,
        so this buffers only the opening of the response (up to the
        longest known refusal prefix) before deciding: if it never matches
        a refusal opener, the buffered tokens are flushed and the rest of
        the stream passes through untouched — no perceptible added
        latency beyond the first ~20 chars. If it does match, the whole
        stream is discarded and restarted fresh, up to `refusal_retries`
        times, before giving up and streaming the refusal through as-is."""
        for attempt in range(refusal_retries + 1):
            buffered: list[str] = []
            buffered_text = ""
            is_refusal_opener = None  # None = undecided, True/False once resolved
            async for token in self.generate_stream(messages, model=model):
                if is_refusal_opener is False:
                    yield token
                    continue
                buffered.append(token)
                buffered_text += token
                stripped = buffered_text.strip().lower()
                if any(stripped.startswith(p) for p in REFUSAL_PREFIXES):
                    is_refusal_opener = True
                elif len(stripped) >= _MAX_REFUSAL_PREFIX_LEN or not any(
                    p.startswith(stripped) for p in REFUSAL_PREFIXES
                ):
                    is_refusal_opener = False
                    for buffered_token in buffered:
                        yield buffered_token

            if is_refusal_opener is None:
                # Stream ended before the buffer resolved either way (a very
                # short response whose entire text still reads as a partial
                # refusal prefix) — fall back to the plain, length-agnostic
                # check against what we actually got.
                is_refusal_opener = is_refusal(buffered_text)
                if not is_refusal_opener:
                    for buffered_token in buffered:
                        yield buffered_token

            if not is_refusal_opener:
                return  # answered normally — already streamed in full

            if attempt < refusal_retries:
                logger.info(f"Stream refusal despite confident retrieval — resampling ({attempt + 1}/{refusal_retries})")
                continue

            # Out of retries — stream the refusal through as-is.
            for buffered_token in buffered:
                yield buffered_token
