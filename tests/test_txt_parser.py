"""
Tests for ingestion/txt_parser.py — plain text ingestion with encoding fallback.
"""
import pytest

from ingestion.txt_parser import TxtParser, decode_text_file


def test_parse_utf8_file(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("Lease Agreement No. 47/2024. This is an English sentence too.", encoding="utf-8")
    parsed = TxtParser().parse(str(p))
    assert parsed.total_pages == 1
    assert parsed.pages[0]["page_num"] == 1
    assert parsed.pages[0]["has_ocr"] is False
    assert "Lease Agreement" in parsed.pages[0]["text"]
    assert parsed.filename == "doc.txt"


def test_parse_utf8_bom_file(tmp_path):
    p = tmp_path / "bom.txt"
    p.write_bytes(b"\xef\xbb\xbfHello world")
    parsed = TxtParser().parse(str(p))
    assert parsed.pages[0]["text"] == "Hello world"  # BOM stripped, not left as a stray char


def test_parse_cp1252_file(tmp_path):
    p = tmp_path / "cp1252.txt"
    text = "The tenant’s obligations — rent, upkeep — are set out below."
    p.write_bytes(text.encode("cp1252"))
    parsed = TxtParser().parse(str(p))
    assert parsed.pages[0]["text"] == text


def test_parse_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        TxtParser().parse("/nonexistent/path/does-not-exist.txt")


def test_parse_empty_file(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    parsed = TxtParser().parse(str(p))
    assert parsed.pages[0]["text"] == ""
    assert parsed.pages[0]["char_count"] == 0


def test_decode_text_file_never_raises_on_arbitrary_bytes():
    garbage = bytes([0xff, 0xfe, 0x00, 0x81, 0x82, 0x83])
    result = decode_text_file(garbage)
    assert isinstance(result, str)
