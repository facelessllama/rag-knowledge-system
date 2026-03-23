"""
Handles communication with LLM backend.

Uses Ollama's OpenAI-compatible API endpoint
so switching to real OpenAI requires only URL change.
"""
import logging
import httpx
import asyncio

logger = logging.getLogger(__name__)


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
                        "answer": data["message"]["content"],
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
                if tokens_yielded > 0:
                    # Partial response already sent — retry would cause duplicates in UI
                    logger.error(f"LLM stream error after {tokens_yielded} tokens, not retrying")
                    raise
                logger.warning(f"LLM stream {type(e).__name__} attempt {attempt}/{retries} (no tokens sent)")
                if attempt < retries:
                    await asyncio.sleep(2 * attempt)
            except Exception as e:
                logger.error(f"LLM stream error: {e}")
                raise
        logger.error(f"LLM stream failed after {retries} attempts")
        raise last_error
