"""
Tests for rag/prompt_builder.py — context assembly, token/char budgets,
multi-doc compare-mode detection.
"""
from rag.prompt_builder import PromptBuilder, MAX_CONTEXT_CHARS, MAX_HISTORY_CHARS


def _chunk(text, filename="a.pdf", page=1):
    return {"text": text, "filename": filename, "page_num": page}


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


def test_build_language_rule_forces_russian():
    pb = PromptBuilder()
    messages = pb.build(query="q", chunks=[_chunk("x")], language="ru")
    assert "Russian" in messages[0]["content"]


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
