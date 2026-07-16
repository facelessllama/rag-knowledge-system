"""
TXT Parser Module
Reads plain text files (RU/EN), producing the same ParsedDocument shape
as pdf_parser.py so the rest of the ingestion pipeline (chunker, embedder,
Qdrant upsert) needs no format-specific handling.
"""
import logging
from pathlib import Path

from ingestion.document import ParsedDocument

logger = logging.getLogger(__name__)

# Windows-originated Russian text files are often cp1251, not UTF-8.
# latin-1 never raises — guaranteed fallback so decoding can't hard-fail.
_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251", "cp1252", "latin-1")


def decode_text_file(raw: bytes) -> str:
    """Best-effort decode trying common encodings, falling back to latin-1
    (which never raises)."""
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


class TxtParser:
    """Parses plain text files. Entire file is treated as a single page."""

    def parse(self, file_path: str) -> ParsedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = decode_text_file(path.read_bytes())
        logger.info(f"Parsed {path.name}: {len(text)} chars")

        return ParsedDocument(
            filename=path.name,
            total_pages=1,
            pages=[{
                "page_num": 1,
                "text": text,
                "has_ocr": False,
                "char_count": len(text),
            }],
            metadata={
                "title": path.stem,
                "author": "Unknown",
                "subject": "",
                "creator": "",
                "page_count": 1,
            },
            file_size_kb=path.stat().st_size / 1024,
        )
