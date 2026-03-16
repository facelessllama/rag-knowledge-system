"""
Prompt Builder
Constructs LLM prompts with retrieved context and citation instructions.
"""
import logging
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an intelligent knowledge base assistant.
Answer questions STRICTLY based on the provided document excerpts.

Rules:
1. Answer ONLY using information from the provided context
2. Do NOT insert source references like [Page X] or [Doc: Y] into your answer text
3. Be concise and precise — 2-5 sentences max
4. If the answer is not in the context, say: "I could not find this information in the provided documents"
5. Use the same language as the question (Russian or English)
6. Never repeat the question back"""


class PromptBuilder:
    def build(
        self,
        query: str,
        chunks: list[dict],
        chat_history: list[dict] = None
    ) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if chat_history:
            for turn in chat_history[-4:]:
                messages.append(turn)

        context = self._format_context(chunks)
        user_message = f"""Context from documents:
{context}

Question: {query}

Answer based on the context above. Do not include page or document references in your answer."""

        messages.append({"role": "user", "content": user_message})
        logger.info(f"Prompt built | chunks={len(chunks)} history={len(chat_history) if chat_history else 0}")
        return messages

    def _format_context(self, chunks: list[dict]) -> str:
        if not chunks:
            return "No relevant context found."

        formatted = []
        for i, chunk in enumerate(chunks, 1):
            page_info = f"Page {chunk.get('page_num', '?')}"
            formatted.append(f"[Excerpt {i} | {page_info}]\n{chunk['text']}")

        return "\n\n".join(formatted)
