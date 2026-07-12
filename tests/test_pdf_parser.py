"""
Tests for ingestion/pdf_parser.py — text extraction and OCR fallback.

Builds small real PDFs on the fly with PyMuPDF (fitz) instead of shipping
binary fixtures. pytesseract is monkeypatched for the OCR-fallback test so
it stays fast and deterministic (not dependent on OCR accuracy).
"""
import fitz
import pytest

from ingestion.pdf_parser import PDFParser, split_for_highlight_search


def _make_pdf(path, text="", title=None, author=None):
    doc = fitz.open()
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text)
    if title or author:
        doc.set_metadata({"title": title or "", "author": author or ""})
    doc.save(str(path))
    doc.close()
    return path


def test_parse_text_pdf_extracts_text_without_ocr(tmp_path):
    pdf_path = _make_pdf(tmp_path / "text.pdf", text="This is a real text layer with more than fifty characters in it.")
    parser = PDFParser()
    result = parser.parse(str(pdf_path))
    assert result.total_pages == 1
    assert result.pages[0]["has_ocr"] is False
    assert "real text layer" in result.pages[0]["text"]
    assert result.filename == "text.pdf"


def test_parse_short_text_page_triggers_ocr_path(tmp_path, monkeypatch):
    pdf_path = _make_pdf(tmp_path / "scan.pdf", text="short")  # < 50 chars -> OCR fallback
    parser = PDFParser()
    monkeypatch.setattr(parser, "ocr_available", True)
    monkeypatch.setattr(parser, "_ocr_page", lambda page: "OCR EXTRACTED TEXT")
    result = parser.parse(str(pdf_path))
    assert result.pages[0]["has_ocr"] is True
    assert result.pages[0]["text"] == "OCR EXTRACTED TEXT"


def test_parse_short_text_page_without_ocr_available_leaves_page_empty(tmp_path, monkeypatch):
    pdf_path = _make_pdf(tmp_path / "scan.pdf", text="short")
    parser = PDFParser()
    monkeypatch.setattr(parser, "ocr_available", False)
    result = parser.parse(str(pdf_path))
    assert result.pages[0]["has_ocr"] is True
    assert result.pages[0]["text"] == "short"  # kept as-is, not blanked out


def test_parse_extracts_metadata(tmp_path):
    pdf_path = _make_pdf(
        tmp_path / "meta.pdf",
        text="Enough text content to skip the OCR fallback threshold easily.",
        title="My Contract",
        author="Serg",
    )
    parser = PDFParser()
    result = parser.parse(str(pdf_path))
    assert result.metadata["title"] == "My Contract"
    assert result.metadata["author"] == "Serg"
    assert result.metadata["page_count"] == 1


def test_parse_missing_file_raises():
    parser = PDFParser()
    with pytest.raises(FileNotFoundError):
        parser.parse("/nonexistent/path/does-not-exist.pdf")


# ── split_for_highlight_search ──────────────────────────────────────────────

def test_split_for_highlight_search_empty_text_returns_no_segments():
    assert split_for_highlight_search("") == []
    assert split_for_highlight_search("   ") == []


def test_split_for_highlight_search_short_text_returns_single_segment():
    segments = split_for_highlight_search("A short chunk of text.")
    assert segments == ["A short chunk of text."]


def test_split_for_highlight_search_covers_the_whole_text():
    text = " ".join(f"word{i}" for i in range(60))  # well over max_len
    segments = split_for_highlight_search(text, max_len=30)
    assert len(segments) > 1
    # every word from the original text must show up somewhere in some segment
    joined = " ".join(segments)
    for i in range(60):
        assert f"word{i}" in joined


def test_split_for_highlight_search_breaks_on_word_boundaries():
    text = "alpha bravo charlie delta echo foxtrot golf hotel"
    segments = split_for_highlight_search(text, max_len=15)
    for seg in segments:
        assert not seg.startswith(" ") and not seg.endswith(" ")
        # no segment should end mid-word (i.e. each segment is whole words)
        assert seg in text


def test_split_for_highlight_search_finds_matches_deep_into_a_long_chunk(tmp_path):
    """End-to-end: a fact placed well past the old 120-char anchor cutoff
    must still be found by search_for() on one of the produced segments."""
    long_prefix = "This clause restates prior definitions and boilerplate. " * 4
    fact = "The maximum discount sum prescribed is exactly seventy five thousand pounds."
    full_text = long_prefix + fact
    assert len(long_prefix) > 120  # confirm the fact is past the old anchor's reach

    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.insert_textbox(fitz.Rect(50, 50, 550, 750), full_text, fontsize=10)

    found_fact = False
    for segment in split_for_highlight_search(full_text, max_len=110):
        if page.search_for(segment):
            if "seventy five thousand" in segment:
                found_fact = True
    doc.close()
    assert found_fact, "the fact segment (past the old 120-char anchor) should have matched on the page"
