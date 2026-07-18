"""
PDF Parser Module
Handles both text-based PDFs and scanned documents (OCR)
"""
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageDraw
import io
import logging
import time
from typing import Optional
from pathlib import Path

from ingestion.document import ParsedDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFParser:
    """
    Parses PDF files extracting text and metadata.
    Falls back to OCR for scanned pages.
    """

    # Total pixels an OCR pixmap is allowed to reach before the zoom factor
    # is scaled down — an unbounded 2x zoom on an oversized page (a scanned
    # poster/map page, or a maliciously crafted huge MediaBox) can demand a
    # multi-hundred-megabyte in-memory bitmap per page; this caps it
    # regardless of the source page's declared dimensions.
    MAX_OCR_PIXELS = 40_000_000  # ~40MP, generous for a single scanned page

    # A page whose native text layer is shorter than this is treated as a
    # candidate for OCR (classic scanned page with no/near-no text layer).
    # Deliberately NOT the only trigger — see MIN_NATIVE_TEXT_CHARS's use
    # alongside image coverage in _needs_ocr().
    MIN_NATIVE_TEXT_CHARS = 50

    # Fraction of the page's area a placed image (or images) must cover
    # before OCR is attempted regardless of how much native text is also on
    # the page. Catches the case a length-only threshold misses entirely: a
    # scanned page carrying a *long* digital header/footer/watermark (well
    # over MIN_NATIVE_TEXT_CHARS) whose actual body content is a scanned
    # image with no text layer at all — the old "len(text) < 50" check would
    # never even look at that page, silently dropping its entire body.
    IMAGE_COVERAGE_OCR_THRESHOLD = 0.5

    def __init__(self, ocr_language: str = "eng", max_pages: int = 500, ocr_timeout_seconds: int = 60,
                 max_total_ocr_seconds: int = 600):
        self.ocr_language = ocr_language
        self.max_pages = max_pages
        self.ocr_timeout_seconds = ocr_timeout_seconds
        # ocr_timeout_seconds bounds a single page's Tesseract call, not the
        # document as a whole — a 500-page scan with every page needing OCR
        # could otherwise legitimately run for `max_pages * ocr_timeout_
        # seconds` (hours). This is a wall-clock budget across the WHOLE
        # parse() call: once exhausted, remaining pages needing OCR are left
        # empty (logged once, not once per page) rather than attempted.
        self.max_total_ocr_seconds = max_total_ocr_seconds
        self.ocr_available = self._check_tesseract()
        logger.info(
            f"PDFParser initialized | OCR language: {ocr_language} | OCR available: {self.ocr_available} "
            f"| max_pages: {max_pages} | ocr_timeout: {ocr_timeout_seconds}s | max_total_ocr: {max_total_ocr_seconds}s"
        )

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

        # Checked before any per-page work (OCR in particular) starts —
        # an oversized document should fail fast, not burn CPU/OCR time on
        # the first few hundred pages before finally being rejected.
        if len(doc) > self.max_pages:
            page_count = len(doc)
            doc.close()
            raise ValueError(f"PDF has {page_count} pages, exceeds the {self.max_pages}-page limit")

        pages = []
        # Set on the FIRST page that actually needs OCR, not at parse()
        # entry — a document with zero scanned pages shouldn't have its
        # (unused) OCR budget ticking against a clock nobody checks.
        ocr_deadline = None
        ocr_budget_exhausted = False

        for page_num in range(len(doc)):
            page = doc[page_num]
            native_text, visible_blocks = self._extract_visible_native_text(page)

            # Evidence-based, not length-only: a page whose body is a
            # scanned image but which also carries a long digital header/
            # footer/watermark (over MIN_NATIVE_TEXT_CHARS) used to never be
            # considered for OCR at all under a native-text-length check
            # alone — its entire body was silently dropped. Image coverage
            # catches that case regardless of how much other text is on the
            # page; the length check on its own still catches the classic
            # near-blank scanned page.
            needs_ocr = (
                len(native_text) < self.MIN_NATIVE_TEXT_CHARS
                or self._image_coverage_ratio(page) >= self.IMAGE_COVERAGE_OCR_THRESHOLD
            )

            ocr_text = ""
            if needs_ocr:
                if self.ocr_available:
                    if ocr_deadline is None:
                        ocr_deadline = time.time() + self.max_total_ocr_seconds
                    if time.time() >= ocr_deadline:
                        if not ocr_budget_exhausted:
                            logger.error(
                                f"OCR time budget ({self.max_total_ocr_seconds}s) exhausted at page "
                                f"{page_num + 1}/{len(doc)} — remaining scanned pages keep native text only"
                            )
                            ocr_budget_exhausted = True
                    else:
                        logger.info(f"Page {page_num + 1}: looks scanned, running OCR...")
                        ocr_text = self._ocr_page(page, visible_blocks)
                        if not ocr_text:
                            logger.error(f"Page {page_num + 1}: OCR returned no text — keeping native text ({len(native_text)} chars)")
                else:
                    logger.error(f"Page {page_num + 1}: looks scanned but Tesseract unavailable — keeping native text ({len(native_text)} chars)")

            # Never silently discard non-trivial native text just because
            # OCR ran (or was warranted but couldn't run) — a page's
            # existing, correct short text used to get overwritten with ""
            # the instant OCR came back empty. `has_ocr` now means "this
            # page's persisted text actually includes real OCR output", not
            # "this page merely looked like a candidate for OCR" — it was
            # previously set to True even when Tesseract was unavailable and
            # no OCR attempt happened at all.
            if ocr_text:
                if len(native_text) >= self.MIN_NATIVE_TEXT_CHARS:
                    # Mixed page: substantial native text (header/footer)
                    # AND a substantial OCR contribution (scanned body) —
                    # keep both instead of one clobbering the other.
                    text = f"{native_text}\n{ocr_text}"
                else:
                    text = ocr_text
                has_ocr = True
            else:
                text = native_text
                has_ocr = False

            pages.append({
                "page_num": page_num + 1,
                "text": text,
                "has_ocr": has_ocr,
                "char_count": len(text)
            })

        metadata = self._extract_metadata(doc, path)
        doc.close()

        ocr_pages = sum(1 for p in pages if p['has_ocr'])
        empty_pages = sum(1 for p in pages if not p['text'])
        logger.info(f"Parsed {path.name}: {len(pages)} pages, {ocr_pages} OCR pages")
        if empty_pages:
            logger.error(f"{path.name}: {empty_pages} page(s) produced no text — they will be absent from search results")

        return ParsedDocument(
            filename=path.name,
            total_pages=len(pages),
            pages=pages,
            metadata=metadata,
            file_size_kb=path.stat().st_size / 1024
        )

    def _extract_visible_native_text(self, page) -> tuple[str, list]:
        """Returns (text, span_bboxes) — VISIBLE text only. Scanned PDFs
        that already went through some prior OCR pass commonly carry that
        OCR's output as an invisible text layer (PDF render mode 3, laid
        transparently over the raster image so the page looks like a plain
        scan but is still selectable/searchable) — PyMuPDF surfaces this as
        `alpha == 0` on the span in `get_text("dict")`. Treating that layer
        as ordinary native text would be wrong two ways: its (possibly bad)
        content could be long enough to suppress a fresh OCR pass entirely
        (MIN_NATIVE_TEXT_CHARS), and even if OCR does run, masking its bbox
        like any other native text (see _ocr_page) would blank out exactly
        the image region a fresh OCR pass needs to inspect to correct it —
        locking in whatever the original (possibly poor-quality) OCR layer
        said, forever. Only VISIBLE spans are returned here, for both the
        length/decision check and the mask-before-fresh-OCR step."""
        text_lines = []
        blocks = []
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                line_parts = []
                for span in line.get("spans", []):
                    if span.get("alpha", 255) == 0:
                        continue  # invisible — likely an existing OCR text layer, not real content
                    span_text = span.get("text", "")
                    if span_text:
                        line_parts.append(span_text)
                        blocks.append(span["bbox"])
                if line_parts:
                    text_lines.append("".join(line_parts))
        return "\n".join(text_lines).strip(), blocks

    def _image_coverage_ratio(self, page) -> float:
        """Fraction of the page's area covered by placed images (0.0-1.0),
        via each image's placement bbox (get_image_info) — computed as the
        area of the UNION of those bboxes, not their naive sum, so that
        several overlapping/stacked images (a common pattern: a base scan
        plus a stamp or redaction image on top) don't inflate the ratio
        past what's actually covered."""
        page_area = page.rect.width * page.rect.height
        if page_area <= 0:
            return 0.0
        rects = []
        for img in page.get_image_info():
            bbox = img.get("bbox")
            if not bbox:
                continue
            x0, y0, x1, y1 = bbox
            x0, x1 = max(x0, page.rect.x0), min(x1, page.rect.x1)
            y0, y1 = max(y0, page.rect.y0), min(y1, page.rect.y1)
            if x1 > x0 and y1 > y0:
                rects.append((x0, y0, x1, y1))
        return max(0.0, min(self._union_area(rects) / page_area, 1.0))

    @staticmethod
    def _union_area(rects: list) -> float:
        """Exact area covered by the union of axis-aligned rectangles, via
        coordinate compression — a handful of images per page is the norm,
        so the O(n^2) cell count here is negligible; a naive sum of
        individual areas would double-count any overlap, and for stacked/
        overlapping images can even exceed the page's own area."""
        if not rects:
            return 0.0
        xs = sorted({r[0] for r in rects} | {r[2] for r in rects})
        ys = sorted({r[1] for r in rects} | {r[3] for r in rects})
        total = 0.0
        for i in range(len(xs) - 1):
            x0, x1 = xs[i], xs[i + 1]
            width = x1 - x0
            if width <= 0:
                continue
            cx = (x0 + x1) / 2
            for j in range(len(ys) - 1):
                y0, y1 = ys[j], ys[j + 1]
                height = y1 - y0
                if height <= 0:
                    continue
                cy = (y0 + y1) / 2
                if any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in rects):
                    total += width * height
        return total

    def _ocr_page(self, page, mask_blocks: Optional[list] = None) -> str:
        """Render the page and run Tesseract OCR on it.

        PyMuPDF also ships a built-in `page.get_textpage_ocr(full=False)`
        that OCRs only regions NOT already covered by legible native text —
        genuinely the cleaner design (no risk of re-recognizing text this
        page's native extraction already got correctly). It was tried here
        first, but MuPDF's Tesseract integration is an in-process linked-
        library call with no timeout parameter at all (`Pixmap.
        pdfocr_tobytes` takes none), unlike `pytesseract.image_to_string
        (timeout=...)`, which genuinely kills the underlying Tesseract
        SUBPROCESS on expiry (see ocr_timeout_seconds below) — a hang in an
        in-process call can't be force-stopped the same way (Python can't
        kill its own thread, only wait on it), which would silently give up
        the one guarantee against a pathological page stalling ingestion (or
        the ingestion semaphore slot it holds) forever. Kept the
        subprocess-based approach for that reason, and mask out `mask_blocks`
        (VISIBLE native text spans only — see _extract_visible_native_text,
        which is what the caller derives this list from) from the rendered
        image before OCR-ing it instead — the same duplication
        `get_textpage_ocr(full=False)` avoids, since `get_pixmap()`
        rasterizes EVERYTHING visible on the page (native text included, not
        just placed images), so an unmasked render lets Tesseract
        re-recognize text the native layer already extracted cleanly,
        producing a near-duplicate of it in the OCR output. Deliberately NOT
        masking invisible-text-layer regions here — those are exactly the
        image areas a fresh OCR pass needs to actually look at (see
        _extract_visible_native_text for why)."""
        try:
            zoom = 2.0  # 2x zoom for better OCR quality
            rect = page.rect
            projected_pixels = (rect.width * zoom) * (rect.height * zoom)
            if projected_pixels > self.MAX_OCR_PIXELS:
                zoom = max(zoom * (self.MAX_OCR_PIXELS / projected_pixels) ** 0.5, 0.5)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            draw = ImageDraw.Draw(img)
            for x0, y0, x1, y1 in (mask_blocks or []):
                draw.rectangle([x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom], fill="white")

            # timeout genuinely kills the underlying tesseract subprocess
            # (pytesseract.timeout_manager) rather than just abandoning a
            # Python-level wait — a hung/pathological page can't stall
            # ingestion (or the ingestion semaphore slot it holds) forever.
            text = pytesseract.image_to_string(img, lang=self.ocr_language, timeout=self.ocr_timeout_seconds)
            return text.strip()
        except RuntimeError as e:
            logger.error(f"OCR timed out after {self.ocr_timeout_seconds}s: {e}")
            return ""
        except Exception as e:
            logger.error(f"OCR failed: {e}")
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
