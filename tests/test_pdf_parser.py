"""
Tests for ingestion/pdf_parser.py — text extraction and OCR fallback.

Builds small real PDFs on the fly with PyMuPDF (fitz) instead of shipping
binary fixtures. pytesseract is monkeypatched for the OCR-fallback test so
it stays fast and deterministic (not dependent on OCR accuracy).
"""
import fitz
import pytest

from ingestion.pdf_parser import PDFParser


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
