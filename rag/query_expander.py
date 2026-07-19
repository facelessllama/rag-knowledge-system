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

    async def contextualize(self, query: str, chat_history: list[dict] | None) -> str:
        """Rewrites a follow-up question into a standalone one using recent
        chat history, so retrieval — which only ever sees the string this
        returns, never chat_history itself (chat_history is passed to
        rag/prompt_builder.py separately, for the final ANSWER generation
        only, after retrieval is already done) — can resolve references
        like "the court" or "this order" to what they actually refer to.

        Without this, a follow-up like "What was the court's final
        decision?" embeds/searches on that literal text alone: generic
        across every court case in the corpus, so hybrid search has no
        signal pointing at the ONE case the user means, and the reranker
        correctly scores every candidate low — verified live, same
        best_rerank_score (-1.55, vs. a 3.0 threshold) whether or not
        chat_history was even included in the request, proving it wasn't
        influencing retrieval at all before this existed.

        No-op (zero LLM calls) when there's no history — the common case
        (a first message) pays nothing extra. Falls back to the original
        query on any failure, same defensive pattern as expand() below: a
        worse retrieval query beats a broken request."""
        if not chat_history:
            return query

        recent = chat_history[-4:]  # last 2 exchanges — enough context, keeps the prompt small
        convo = "\n".join(f"{t.get('role', '')}: {t.get('content', '')}" for t in recent)
        prompt = f"""Conversation so far:
{convo}

Latest question: {query}

Rewrite the latest question as a standalone search query that spells out any case, document, or subject implied by the conversation (e.g. replace "the court" or "this order" with what they actually refer to). If the question is already standalone, return it unchanged.

Rules:
- Output ONLY the rewritten question, nothing else
- No quotes, no explanation, one line

Standalone question:"""

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "think": False,
                        "options": {"temperature": 0.1, "num_predict": 60}
                    }
                )
                response.raise_for_status()
                data = response.json()
                rewritten = data["message"]["content"].strip().strip('"').strip()

                if not rewritten or len(rewritten) < 3:
                    return query

                if rewritten.lower() != query.lower():
                    logger.info(f"Contextualized follow-up: {query!r} -> {rewritten!r}")
                return rewritten

        except Exception as e:
            logger.warning(f"Query contextualization failed: {e} — using original question")
            return query

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
- Each query must be short and focused on ONE topic

Examples:
Input: "what's the late payment penalty and how much is the rent?"
Output:
late payment penalty amount
fee for delayed payment
rental cost of the premises
rent price

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
                        "think": False,  # reasoning models would blow the 80-token budget on <think> alone
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

                # Always include the original first
                all_queries = [query]
                for line in lines:
                    if line.lower() != query.lower() and len(line) > 3:
                        all_queries.append(line)

                # Deduplicate
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
