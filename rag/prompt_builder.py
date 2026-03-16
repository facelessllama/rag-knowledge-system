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

MULTI_DOC_ADDITION = """
7. If context contains excerpts from MULTIPLE documents — compare them and highlight any differences or contradictions between documents
8. Structure your answer by document when comparing: first summarize what Document A says, then Document B"""


class PromptBuilder:
    def build(
        self,
        query: str,
        chunks: list[dict],
        chat_history: list[dict] = None
    ) -> list[dict]:
        # Определяем количество уникальных документов
        unique_docs = set(c.get('filename', '') for c in chunks if c.get('filename'))
        is_multi_doc = len(unique_docs) > 1

        system = SYSTEM_PROMPT
        if is_multi_doc:
            system += MULTI_DOC_ADDITION
            logger.info(f"Multi-doc mode: {unique_docs}")

        messages = [{"role": "system", "content": system}]

        if chat_history:
            for turn in chat_history[-4:]:
                messages.append(turn)

        context = self._format_context(chunks, is_multi_doc)
        user_message = f"""Context from documents:
{context}

Question: {query}

Answer based on the context above.{' Compare documents if they contain different information.' if is_multi_doc else ''}"""

        messages.append({"role": "user", "content": user_message})
        logger.info(f"Prompt built | chunks={len(chunks)} history={len(chat_history) if chat_history else 0} multi_doc={is_multi_doc}")
        return messages

    def _format_context(self, chunks: list[dict], is_multi_doc: bool = False) -> str:
        if not chunks:
            return "No relevant context found."

        formatted = []
        for i, chunk in enumerate(chunks, 1):
            page_info = f"Page {chunk.get('page_num', '?')}"
            filename = chunk.get('filename', '')

            if is_multi_doc and filename:
                # Показываем имя документа в контексте
                label = f"[{filename} | {page_info}]"
            else:
                label = f"[Excerpt {i} | {page_info}]"

            formatted.append(f"{label}\n{chunk['text']}")

        return "\n\n".join(formatted)
