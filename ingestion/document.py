"""
Shared parsed-document result type — produced by any format-specific parser
(pdf_parser.py, txt_parser.py, ...) and consumed identically by chunker.py.
"""
from dataclasses import dataclass


@dataclass
class ParsedDocument:
    """Structured output from document parsing"""
    filename: str
    total_pages: int
    pages: list[dict]  # [{page_num, text, has_ocr, char_count}]
    metadata: dict
    file_size_kb: float
