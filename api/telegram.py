"""
Telegram Bot Integration
========================
Demonstrates how any external system can integrate with the RAG service via webhook.

Setup:
1. Create a bot via @BotFather, get TELEGRAM_BOT_TOKEN
2. Add the token to .env
3. Register the webhook:
   curl -X POST "https://api.telegram.org/bot<TOKEN>/setWebhook" \
        -d "url=https://your-domain.com/telegram/webhook"
   (for local development, use ngrok or cloudflared)
"""
import os
import logging
import httpx
from fastapi import APIRouter, Request, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
API_KEY = os.getenv("API_KEY", "")
INTERNAL_API_URL = os.getenv("INTERNAL_API_URL", "http://localhost:8000")


@router.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    """Receives messages from Telegram and replies via the RAG service."""
    # Verify webhook secret token if configured
    if TELEGRAM_WEBHOOK_SECRET:
        token = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if token != TELEGRAM_WEBHOOK_SECRET:
            raise HTTPException(403, "Invalid webhook secret")

    data = await request.json()
    logger.info(f"Telegram update: {data}")

    message = data.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "").strip()

    if not chat_id or not text or text.startswith("/"):
        return {"ok": True}

    answer = await _get_rag_answer(text)
    await _send_message(chat_id, answer)
    return {"ok": True}


async def _get_rag_answer(question: str) -> str:
    try:
        headers = {"Content-Type": "application/json"}
        if API_KEY:
            headers["X-API-Key"] = API_KEY

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{INTERNAL_API_URL}/query",
                json={"question": question, "top_k": 3, "channel": "telegram"},
                headers=headers
            )
            result = response.json()
            answer = result.get("answer", "Could not get an answer.")

            sources = result.get("sources", [])
            if sources:
                names = list({s.get("filename", "") for s in sources if s.get("filename")})
                if names:
                    answer += f"\n\n📎 Sources: {', '.join(names)}"

            return answer
    except Exception as e:
        logger.error(f"RAG error: {e}")
        return "An error occurred while processing the request."


async def _send_message(chat_id: int, text: str):
    if not TELEGRAM_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set, skipping send")
        return
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text}
        )
        logger.info(f"Telegram send: {resp.status_code}")
