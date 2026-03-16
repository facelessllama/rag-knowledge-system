import httpx
import logging
from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)
router = APIRouter()

BITRIX_WEBHOOK_URL = "https://b24-typ264.bitrix24.ru/rest/1/1zjpjhcsq6ddbodn/"
BOT_ID = 16


@router.post("/bitrix/webhook")
async def bitrix_webhook(request: Request):
    try:
        data = dict(await request.form())
        logger.info(f"Bitrix incoming: {data}")

        event = data.get("event", "")
        if event != "ONIMBOTMESSAGEADD":
            return {"status": "ignored"}

        message_text = data.get("data[PARAMS][MESSAGE]", "").strip()
        dialog_id = data.get("data[PARAMS][DIALOG_ID]", "")

        if not message_text or not dialog_id:
            return {"status": "error", "detail": "missing fields"}

        logger.info(f"User asks: '{message_text}' in dialog {dialog_id}")

        rag_answer = await get_rag_answer(message_text)
        await send_bitrix_message(dialog_id, rag_answer)

        return {"status": "ok"}

    except Exception as e:
        logger.error(f"Bitrix webhook error: {e}", exc_info=True)
        return {"status": "error", "detail": str(e)}


async def get_rag_answer(question: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "http://localhost:8000/query",
                json={"question": question, "top_k": 3}
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
        logger.error(f"RAG query error: {e}")
        return "Извините, произошла ошибка при обработке запроса."


async def send_bitrix_message(dialog_id: str, message: str):
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{BITRIX_WEBHOOK_URL}imbot.message.add.json",
            json={
                "BOT_ID": BOT_ID,
                "DIALOG_ID": dialog_id,
                "MESSAGE": message,
            }
        )
        logger.info(f"Bitrix send: {resp.status_code} {resp.text}")


@router.get("/bitrix/install")
@router.post("/bitrix/install")
async def bitrix_install(request: Request):
    """Endpoint для установки локального приложения Битрикс24"""
    logger.info("Bitrix install called")
    return {"status": "ok", "message": "RAG Bot installed"}
