"""
PDF Parser Module
Handles both text-based PDFs and scanned documents (OCR)
"""
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import logging
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    """Structured output from PDF parsing"""
    filename: str
    total_pages: int
    pages: list[dict]  # [{page_num, text, has_ocr}]
    metadata: dict
    file_size_kb: float


class PDFParser:
    """
    Parses PDF files extracting text and metadata.
    Falls back to OCR for scanned pages.
    """

    def __init__(self, ocr_language: str = "rus+eng"):
        self.ocr_language = ocr_language
        self.ocr_available = self._check_tesseract()
        logger.info(f"PDFParser initialized | OCR language: {ocr_language} | OCR available: {self.ocr_available}")

    def _check_tesseract(self) -> bool:
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            logger.error(f"Tesseract not available — scanned PDFs will produce empty pages: {e}")
            return False

    def parse(self, file_path: str) -> ParsedDocument:
        """Main entry point — parse a PDF file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Parsing: {path.name}")

        doc = fitz.open(file_path)
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            has_ocr = False

            # If page has no text — it's likely a scan, run OCR
            if len(text) < 50:
                logger.info(f"Page {page_num + 1}: no text found, running OCR...")
                text = self._ocr_page(page)
                has_ocr = True

            pages.append({
                "page_num": page_num + 1,
                "text": text,
                "has_ocr": has_ocr,
                "char_count": len(text)
            })

        metadata = self._extract_metadata(doc, path)
        doc.close()

        logger.info(
            f"Parsed {path.name}: {len(pages)} pages, "
            f"{sum(1 for p in pages if p['has_ocr'])} OCR pages"
        )

        return ParsedDocument(
            filename=path.name,
            total_pages=len(pages),
            pages=pages,
            metadata=metadata,
            file_size_kb=path.stat().st_size / 1024
        )

    def _ocr_page(self, page) -> str:
        """Convert page to image and run Tesseract OCR"""
        try:
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            text = pytesseract.image_to_string(img, lang=self.ocr_language)
            return text.strip()
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ""

    def _extract_metadata(self, doc, path: Path) -> dict:
        """Extract document metadata"""
        meta = doc.metadata or {}
        return {
            "title": meta.get("title", path.stem),
            "author": meta.get("author", "Unknown"),
            "subject": meta.get("subject", ""),
            "creator": meta.get("creator", ""),
            "page_count": len(doc),
            "file_size_kb": round(path.stat().st_size / 1024, 2)
        }
