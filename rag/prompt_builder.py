"""
Prompt Builder
Constructs LLM prompts with retrieved context and citation instructions.
"""
import logging
logger = logging.getLogger(__name__)

# Character budgets (1 token ≈ 4 chars; qwen2.5:7b has 32k token context)
# Reserve ~1500 tokens for system prompt + question + answer overhead.
MAX_CONTEXT_CHARS = 12_000   # ~3000 tokens for retrieved chunks
MAX_HISTORY_CHARS = 8_000    # ~2000 tokens for chat history turns


SYSTEM_PROMPT = """You are an intelligent knowledge base assistant.
Answer questions STRICTLY based on the provided document excerpts.

Rules:
1. Answer ONLY using information from the provided context
2. Do NOT insert source references like [Page X] or [Doc: Y] into your answer text
3. Be concise and precise — 2-5 sentences max
4. Interpret context broadly: if the answer is implied or uses different wording (e.g. "interest charge" answers a question about "penalty"), use it. Only say "I could not find this information in the provided documents" if the context is truly unrelated to the question.
5. {lang_rule}
6. Never repeat the question back"""

LANG_RULES = {
    "en": "Always respond in English, regardless of the language of the question or documents.",
    "ru": "Always respond in Russian, regardless of the language of the question or documents.",
    None:  "Use the same language as the question (Russian or English)",
}

MULTI_DOC_ADDITION = """
7. If context contains excerpts from MULTIPLE documents — compare them and highlight any differences or contradictions between documents
8. Structure your answer by document when comparing: first summarize what Document A says, then Document B"""

TELEGRAM_SYSTEM_PROMPT = """You are a knowledge base assistant in a Telegram chat.
Answer questions STRICTLY based on the provided document excerpts.

Rules:
1. Answer ONLY using information from the provided context
2. Be very brief — 1-3 sentences maximum, no lists, no headers
3. Plain text only — no markdown, no asterisks, no formatting
4. If the answer is not in the context, say: "Не нашёл информацию по этому вопросу в базе знаний."
5. {lang_rule}
6. Never repeat the question back"""


class PromptBuilder:
    def build(
        self,
        query: str,
        chunks: list[dict],
        chat_history: list[dict] = None,
        language: str = None,
        channel: str = None,
    ) -> list[dict]:
        unique_docs = set(c.get('filename', '') for c in chunks if c.get('filename'))
        is_multi_doc = len(unique_docs) > 1

        lang_rule = LANG_RULES.get(language, LANG_RULES[None])
        if channel == "telegram":
            system = TELEGRAM_SYSTEM_PROMPT.format(lang_rule=lang_rule)
        else:
            system = SYSTEM_PROMPT.format(lang_rule=lang_rule)
            if is_multi_doc:
                system += MULTI_DOC_ADDITION
                logger.info(f"Multi-doc mode: {unique_docs}")

        messages = [{"role": "system", "content": system}]

        # Trim chat history to budget (drop oldest turns first)
        if chat_history:
            trimmed_history = self._trim_history(chat_history)
            for turn in trimmed_history:
                messages.append(turn)

        context = self._format_context(chunks, is_multi_doc)
        user_message = f"""Context from documents:
{context}

Question: {query}

Answer based on the context above.{' Compare documents if they contain different information.' if is_multi_doc else ''}"""

        messages.append({"role": "user", "content": user_message})
        logger.info(f"Prompt built | chunks={len(chunks)} history={len(chat_history) if chat_history else 0} multi_doc={is_multi_doc}")
        return messages

    def _trim_history(self, history: list[dict]) -> list[dict]:
        """Keep the last N turns that fit within MAX_HISTORY_CHARS, always keeping pairs."""
        # Work with last 4 turns max, then apply char budget
        recent = history[-4:]
        total = sum(len(t.get("content", "")) for t in recent)
        while total > MAX_HISTORY_CHARS and len(recent) >= 2:
            # Drop oldest pair (user + assistant)
            dropped = recent[:2]
            total -= sum(len(t.get("content", "")) for t in dropped)
            recent = recent[2:]
            logger.info(f"History trimmed: dropped 2 turns ({sum(len(t.get('content','')) for t in dropped)} chars), {total} chars remaining")
        return recent

    def _format_context(self, chunks: list[dict], is_multi_doc: bool = False) -> str:
        if not chunks:
            return "No relevant context found."

        formatted = []
        total_chars = 0
        for i, chunk in enumerate(chunks, 1):
            page_info = f"Page {chunk.get('page_num', '?')}"
            filename = chunk.get('filename', '')

            if is_multi_doc and filename:
                label = f"[{filename} | {page_info}]"
            else:
                label = f"[Excerpt {i} | {page_info}]"

            entry = f"{label}\n{chunk['text']}"

            if total_chars + len(entry) > MAX_CONTEXT_CHARS:
                logger.warning(f"Context budget reached at chunk {i}/{len(chunks)} — truncating")
                break

            formatted.append(entry)
            total_chars += len(entry)

        return "\n\n".join(formatted)
