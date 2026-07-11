"""
Tests for rag/generator.py — leaked <question> tag stripping.
Some models (observed on qwen3) occasionally echo the <question> wrapper
used to isolate user input instead of just answering.
"""
from rag.generator import strip_leaked_question_tag


def test_strips_leaked_question_tag_prefix():
    text = "<question>Where is it banned?</question> It is banned in several regions."
    assert strip_leaked_question_tag(text) == "It is banned in several regions."


def test_leaves_normal_answer_untouched():
    text = "It is banned in several regions."
    assert strip_leaked_question_tag(text) == text


def test_only_strips_leading_occurrence():
    text = "<question>Q</question> Answer mentions <question>nested</question> literally."
    result = strip_leaked_question_tag(text)
    assert result == "Answer mentions <question>nested</question> literally."


def test_multiline_question_tag_content():
    text = "<question>Line one\nLine two?</question>\nActual answer here."
    assert strip_leaked_question_tag(text) == "Actual answer here."
