"""
Telegram Bot Integration
========================
Демонстрирует, как любая внешняя система может интегрироваться с RAG через webhook.

Настройка:
1. Создать бота через @BotFather, получить TELEGRAM_BOT_TOKEN
2. Добавить токен в .env
3. Зарегистрировать webhook:
   curl -X POST "https://api.telegram.org/bot<TOKEN>/setWebhook" \
        -d "url=https://your-domain.com/telegram/webhook"
   (для локальной разработки используйте ngrok или cloudflared)
"""
import os
import logging
import httpx
from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)
router = APIRouter()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


@router.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    """Принимает сообщения от Telegram и отвечает через RAG."""
    data = await request.json()
    logger.info(f"Telegram update: {data}")

    message = data.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "").strip()

    if not chat_id or not text or text.startswith("/"):
        # Игнорируем команды и пустые сообщения
        return {"ok": True}

    answer = await _get_rag_answer(text)
    await _send_message(chat_id, answer)
    return {"ok": True}


async def _get_rag_answer(question: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "http://localhost:8000/query",
                json={"question": question, "top_k": 3, "channel": "telegram"}
            )
            result = response.json()
            answer = result.get("answer", "Не удалось получить ответ.")

            sources = result.get("sources", [])
            if sources:
                names = list({s.get("filename", "") for s in sources if s.get("filename")})
                if names:
                    answer += f"\n\n📎 Источники: {', '.join(names)}"

            return answer
    except Exception as e:
        logger.error(f"RAG error: {e}")
        return "Произошла ошибка при обработке запроса."


async def _send_message(chat_id: int, text: str):
    if not TELEGRAM_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN не задан, отправка пропущена")
        return
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text}
        )
        logger.info(f"Telegram send: {resp.status_code}")
