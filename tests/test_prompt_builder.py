"""
Tests for rag/prompt_builder.py — context assembly, token/char budgets,
multi-doc compare-mode detection.
"""
from rag.prompt_builder import PromptBuilder, MAX_CONTEXT_CHARS, MAX_HISTORY_CHARS


def _chunk(text, filename="a.pdf", page=1, document_id=None, chunk_index=None):
    c = {"text": text, "filename": filename, "page_num": page}
    if document_id is not None:
        c["document_id"] = document_id
    if chunk_index is not None:
        c["chunk_index"] = chunk_index
    return c


def test_build_single_doc_uses_default_system_prompt():
    pb = PromptBuilder()
    messages = pb.build(query="What is clause 3?", chunks=[_chunk("Clause 3 says X.")])
    assert messages[0]["role"] == "system"
    assert "compare" not in messages[0]["content"].lower()
    assert messages[-1]["role"] == "user"
    assert "<question>What is clause 3?</question>" in messages[-1]["content"]


def test_build_multi_doc_switches_to_compare_prompt():
    pb = PromptBuilder()
    chunks = [_chunk("Doc A says X.", filename="a.pdf"), _chunk("Doc B says Y.", filename="b.pdf")]
    messages = pb.build(query="Compare them", chunks=chunks)
    assert "compare" in messages[0]["content"].lower()
    assert "[a.pdf | Page 1]" in messages[-1]["content"]
    assert "[b.pdf | Page 1]" in messages[-1]["content"]


def test_build_always_forces_english_response():
    pb = PromptBuilder()
    messages = pb.build(query="q", chunks=[_chunk("x")])
    assert "Always respond in English" in messages[0]["content"]


def test_build_telegram_channel_uses_brief_prompt():
    pb = PromptBuilder()
    messages = pb.build(query="q", chunks=[_chunk("x")], channel="telegram")
    assert "Telegram" in messages[0]["content"]
    assert "1-3 sentences" in messages[0]["content"]


def test_context_budget_truncates_when_chunks_exceed_max_chars():
    pb = PromptBuilder()
    big_chunk_text = "x" * (MAX_CONTEXT_CHARS // 2 + 100)
    chunks = [_chunk(big_chunk_text, page=i) for i in range(1, 5)]
    messages = pb.build(query="q", chunks=chunks)
    user_content = messages[-1]["content"]
    # Only the first couple of oversized chunks should fit before the budget cuts off
    assert user_content.count("Excerpt") < len(chunks)


def test_history_trim_keeps_last_turns_within_budget():
    pb = PromptBuilder()
    history = [
        {"role": "user", "content": "x" * 100},
        {"role": "assistant", "content": "y" * 100},
        {"role": "user", "content": "z" * (MAX_HISTORY_CHARS + 500)},
        {"role": "assistant", "content": "w" * 100},
    ]
    messages = pb.build(query="q", chunks=[_chunk("ctx")], chat_history=history)
    history_in_messages = [m for m in messages if m["role"] in ("user", "assistant") and m is not messages[-1]]
    total_chars = sum(len(m["content"]) for m in history_in_messages)
    assert total_chars <= MAX_HISTORY_CHARS + len("w" * 100)  # last pair always kept even if it alone exceeds budget


def test_no_context_returns_placeholder():
    pb = PromptBuilder()
    messages = pb.build(query="q", chunks=[])
    assert "No relevant context found." in messages[-1]["content"]


# ── reading-order reassembly ──────────────────────────────────────────────────

def test_format_context_reorders_chunks_into_reading_order_not_rerank_order():
    """A document's caption chunk (page 1, chunk_index 0) can rank below
    later procedural chunks — presented out of order, a small model can fail
    to connect them (see _order_for_reading's docstring / eval/README.md).
    Chunks must appear in page_num/chunk_index order, not input order."""
    pb = PromptBuilder()
    procedural = _chunk("procedural filler", document_id="d1", page=2, chunk_index=3)
    caption = _chunk("IN THE HIGH COURT... case caption", document_id="d1", page=1, chunk_index=0)
    # Reranker/promotion order: procedural (higher score) before caption (lower score)
    messages = pb.build(query="q", chunks=[procedural, caption])
    user_content = messages[-1]["content"]
    assert user_content.index("case caption") < user_content.index("procedural filler")


def test_format_context_keeps_documents_in_original_relevance_order():
    """Only chunks WITHIN a document get reordered — which document comes
    first must still follow the input (relevance-ranked) order."""
    pb = PromptBuilder()
    chunks = [
        _chunk("doc B chunk", filename="b.pdf", document_id="d2", page=1, chunk_index=0),
        _chunk("doc A late chunk", filename="a.pdf", document_id="d1", page=3, chunk_index=5),
        _chunk("doc A early chunk", filename="a.pdf", document_id="d1", page=1, chunk_index=0),
    ]
    messages = pb.build(query="q", chunks=chunks)
    user_content = messages[-1]["content"]
    # doc B (first in input) stays first; within doc A, early chunk comes before late chunk
    assert user_content.index("doc B chunk") < user_content.index("doc A early chunk")
    assert user_content.index("doc A early chunk") < user_content.index("doc A late chunk")
