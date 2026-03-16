"""
Query Expander
Uses LLM to generate alternative phrasings of the user query.
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
        """
        Generate 2 alternative phrasings + original.
        Returns list of up to 3 queries total.
        On any error — returns just original query.
        """
        prompt = f"""Rephrase the following search query in 2 different ways.
Keep the EXACT same meaning and topic. Only change the wording.
Do NOT change the subject (if query is about kettle, keep it about kettle).
Output ONLY the 2 rephrased queries, one per line, nothing else.

Query: {query}

Rephrased:"""

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {"temperature": 0.2, "num_predict": 60}
                    }
                )
                response.raise_for_status()
                data = response.json()
                text = data["message"]["content"].strip()

                alternatives = [
                    line.strip().lstrip("0123456789.-) ")
                    for line in text.split("\n")
                    if line.strip() and len(line.strip()) > 5
                ][:2]

                all_queries = [query]
                for alt in alternatives:
                    if alt.lower() != query.lower() and len(alt) > 5:
                        all_queries.append(alt)

                logger.info(f"Query expanded: {len(all_queries)} variants | {all_queries}")
                return all_queries

        except Exception as e:
            logger.warning(f"Query expansion failed: {e} — using original")
            return [query]
