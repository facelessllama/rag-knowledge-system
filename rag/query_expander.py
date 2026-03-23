"""
Query Expander
Uses LLM to generate alternative phrasings and decompose complex queries.
"""
import logging
import httpx

logger = logging.getLogger(__name__)


class QueryExpander:
    def __init__(self, ollama_url: str = "http://localhost:11435", model: str = "qwen2.5:7b"):
        self.ollama_url = ollama_url
        self.model = model
        logger.info("QueryExpander ready")

    async def expand(self, query: str) -> list[str]:
        # Fast path: short or single-word queries don't benefit from expansion
        words = query.strip().split()
        if len(words) <= 2:
            logger.info(f"Query expansion skipped (short query): {query!r}")
            return [query]

        prompt = f"""Analyze this search query and do TWO things:
1. If the query contains MULTIPLE questions or topics — split into separate simple queries
2. For each query (original or split) — add 1 rephrased version

Rules:
- Output ONLY queries, one per line
- Maximum 4 lines total
- No numbering, no explanations
- Keep original language (Russian or English)
- Each query must be short and focused on ONE topic

Examples:
Input: "какой размер пени и сколько стоит аренда?"
Output:
размер пени при просрочке оплаты
штраф за задержку платежа
стоимость аренды помещения
цена аренды

Input: "how does the kettle work?"
Output:
how does the kettle work
kettle operation instructions

Query: {query}
Output:"""

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 80}
                    }
                )
                response.raise_for_status()
                data = response.json()
                text = data["message"]["content"].strip()

                lines = [
                    line.strip().lstrip("0123456789.-) ")
                    for line in text.split("\n")
                    if line.strip() and len(line.strip()) > 3
                ][:4]

                # Всегда включаем оригинал первым
                all_queries = [query]
                for line in lines:
                    if line.lower() != query.lower() and len(line) > 3:
                        all_queries.append(line)

                # Дедупликация
                seen = set()
                unique = []
                for q in all_queries:
                    if q.lower() not in seen:
                        seen.add(q.lower())
                        unique.append(q)

                logger.info(f"Query expanded: {len(unique)} variants | {unique}")
                return unique[:5]

        except Exception as e:
            logger.warning(f"Query expansion failed: {e} — using original")
            return [query]
