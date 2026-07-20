"""
Tests for ingestion/pdf_parser.py — text extraction and OCR fallback.

Builds small real PDFs on the fly with PyMuPDF (fitz) instead of shipping
binary fixtures. pytesseract is monkeypatched for the OCR-fallback test so
it stays fast and deterministic (not dependent on OCR accuracy).
"""
import io

import fitz
import pytest
from PIL import Image, ImageDraw

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


def _make_pdf_with_full_page_image(path, header_footer_text=""):
    """A page whose body is a placed image covering (almost) the whole
    page — the scanned-page shape — optionally with digital text on top
    (simulating a long header/footer/watermark over a scanned body)."""
    doc = fitz.open()
    page = doc.new_page()
    img = Image.new("RGB", (600, 800), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(page.rect, stream=buf.getvalue())
    if header_footer_text:
        page.insert_text((72, 30), header_footer_text)
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
    pdf_path = _make_pdf(tmp_path / "scan.pdf", text="short")  # < 50 chars -> OCR candidate
    parser = PDFParser()
    monkeypatch.setattr(parser, "ocr_available", True)
    monkeypatch.setattr(parser, "_ocr_page", lambda page, blocks=None: "OCR EXTRACTED TEXT")
    result = parser.parse(str(pdf_path))
    assert result.pages[0]["has_ocr"] is True
    assert result.pages[0]["text"] == "OCR EXTRACTED TEXT"


def test_parse_short_native_text_kept_when_ocr_unavailable(tmp_path, monkeypatch):
    """OCR never runs at all here (Tesseract unavailable) — has_ocr must be
    False, not True: it now means "this page's text actually includes real
    OCR output", not "this page merely looked like an OCR candidate"."""
    pdf_path = _make_pdf(tmp_path / "scan.pdf", text="short")
    parser = PDFParser()
    monkeypatch.setattr(parser, "ocr_available", False)
    result = parser.parse(str(pdf_path))
    assert result.pages[0]["has_ocr"] is False
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


def test_parse_rejects_pdf_over_page_limit(tmp_path):
    doc = fitz.open()
    for _ in range(3):
        doc.new_page()
    pdf_path = tmp_path / "toobig.pdf"
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser(max_pages=2)
    with pytest.raises(ValueError, match="3 pages.*2-page limit"):
        parser.parse(str(pdf_path))


def test_parse_within_page_limit_succeeds(tmp_path):
    pdf_path = _make_pdf(tmp_path / "ok.pdf", text="Enough text content to skip the OCR fallback threshold easily.")
    parser = PDFParser(max_pages=2)
    result = parser.parse(str(pdf_path))
    assert result.total_pages == 1


def test_ocr_timeout_keeps_native_text_instead_of_blanking_it(tmp_path, monkeypatch):
    """OCR ran, failed (timeout), and returned "" — the page's own short but
    valid native text must survive, not be overwritten with an empty
    string, which is exactly the second bug this rewrite fixes."""
    pdf_path = _make_pdf(tmp_path / "scan.pdf", text="short")
    parser = PDFParser(ocr_timeout_seconds=5)
    monkeypatch.setattr(parser, "ocr_available", True)

    def _raise_timeout(*args, **kwargs):
        raise RuntimeError("Tesseract process timeout")

    monkeypatch.setattr("ingestion.pdf_parser.pytesseract.image_to_string", _raise_timeout)
    result = parser.parse(str(pdf_path))
    assert result.pages[0]["text"] == "short"  # native text preserved, not blanked
    assert result.pages[0]["has_ocr"] is False  # OCR contributed nothing


def test_parse_stops_ocr_after_total_time_budget_exhausted(tmp_path, monkeypatch):
    doc = fitz.open()
    for _ in range(3):
        doc.new_page()  # no text inserted -> every page is an OCR candidate
    pdf_path = tmp_path / "scan3.pdf"
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser(max_total_ocr_seconds=0)  # budget already spent before the first check
    monkeypatch.setattr(parser, "ocr_available", True)
    calls = []
    monkeypatch.setattr(parser, "_ocr_page", lambda page, blocks=None: calls.append(1) or "OCR TEXT")

    result = parser.parse(str(pdf_path))
    assert calls == []  # zero-second budget -> no page ever gets a real OCR attempt
    assert all(p["text"] == "" for p in result.pages)
    assert not any(p["has_ocr"] for p in result.pages)  # OCR never actually ran -> never True


def test_parse_triggers_ocr_via_image_coverage_despite_long_native_text(tmp_path, monkeypatch):
    """The core bug: a page with a long digital header (well over
    MIN_NATIVE_TEXT_CHARS) sitting on top of a full-page scanned image used
    to never be considered for OCR at all under a text-length-only check —
    its entire body was silently dropped. Image coverage must trigger OCR
    here regardless of native text length, and BOTH texts must survive."""
    header = "CONFIDENTIAL — Internal Report — Distribution restricted to authorized personnel only — Page 1"
    assert len(header) >= PDFParser.MIN_NATIVE_TEXT_CHARS
    pdf_path = _make_pdf_with_full_page_image(tmp_path / "mixed.pdf", header_footer_text=header)

    parser = PDFParser()
    monkeypatch.setattr(parser, "ocr_available", True)
    monkeypatch.setattr(parser, "_ocr_page", lambda page, blocks=None: "Scanned body content extracted by OCR.")

    result = parser.parse(str(pdf_path))
    assert result.pages[0]["has_ocr"] is True
    assert "CONFIDENTIAL" in result.pages[0]["text"]
    assert "Scanned body content" in result.pages[0]["text"]


def test_parse_does_not_run_ocr_for_normal_page_with_small_image(tmp_path, monkeypatch):
    """A page that's mostly native text with a small inset figure shouldn't
    be treated as a scan — image coverage well under the threshold."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Plenty of normal native text content on this page, well over the threshold.")
    img = Image.new("RGB", (50, 50), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(fitz.Rect(72, 700, 122, 750), stream=buf.getvalue())
    pdf_path = tmp_path / "small_figure.pdf"
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    calls = []
    monkeypatch.setattr(parser, "ocr_available", True)
    monkeypatch.setattr(parser, "_ocr_page", lambda page, blocks=None: calls.append(1) or "SHOULD NOT BE CALLED")

    result = parser.parse(str(pdf_path))
    assert calls == []
    assert result.pages[0]["has_ocr"] is False


def test_image_coverage_ratio_near_one_for_full_page_image():
    doc = fitz.open()
    page = doc.new_page()
    img = Image.new("RGB", (600, 800), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(page.rect, stream=buf.getvalue())

    parser = PDFParser()
    assert parser._image_coverage_ratio(page) > 0.9


def test_image_coverage_ratio_uses_union_not_naive_sum_for_overlapping_images():
    """Two images stacked over the SAME full-page region (a scan plus a
    stamp/redaction image on top, say) must not push coverage past ~1.0 —
    a naive sum of individual areas would double it (each already ~1.0 on
    its own), which could even exceed the page's own area."""
    doc = fitz.open()
    page = doc.new_page()
    img = Image.new("RGB", (600, 800), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(page.rect, stream=buf.getvalue())
    page.insert_image(page.rect, stream=buf.getvalue())  # second image, same full-page bbox

    parser = PDFParser()
    ratio = parser._image_coverage_ratio(page)
    assert 0.9 < ratio <= 1.0  # union, not ~2.0 from a naive sum


def test_union_area_of_non_overlapping_rects_is_their_sum():
    rects = [(0, 0, 10, 10), (20, 20, 30, 30)]
    assert PDFParser._union_area(rects) == 200.0


def test_union_area_of_fully_overlapping_rects_is_not_double_counted():
    rects = [(0, 0, 10, 10), (0, 0, 10, 10)]
    assert PDFParser._union_area(rects) == 100.0


def test_image_coverage_ratio_zero_with_no_images():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Just text, no images at all on this page.")

    parser = PDFParser()
    assert parser._image_coverage_ratio(page) == 0.0


def test_ocr_page_masks_native_text_regions_before_ocr(tmp_path, monkeypatch):
    """get_pixmap() rasterizes the WHOLE page, native text included — an
    unmasked render lets Tesseract re-recognize text the native extraction
    already got correctly, producing a near-duplicate in the OCR output.
    _ocr_page() must paint over each mask_blocks bbox (as parse() derives
    them via _extract_visible_native_text) before handing the image to
    pytesseract."""
    doc = fitz.open()
    page = doc.new_page()
    img = Image.new("RGB", (600, 800), color="black")  # anything un-masked stays black
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(page.rect, stream=buf.getvalue())
    page.insert_text((72, 30), "A native text header over fifty characters long to be masked out.")

    captured = {}

    def _fake_image_to_string(image, **kwargs):
        captured["image"] = image
        return "irrelevant"

    monkeypatch.setattr("ingestion.pdf_parser.pytesseract.image_to_string", _fake_image_to_string)

    parser = PDFParser()
    _, visible_blocks = parser._extract_visible_native_text(page)
    parser._ocr_page(page, visible_blocks)

    assert "image" in captured
    zoom = 2.0
    # A point inside the text block's bbox (scaled by the render zoom) must
    # be painted white...
    x0, y0, x1, y1 = visible_blocks[0]
    mid_x, mid_y = int((x0 + x1) / 2 * zoom), int((y0 + y1) / 2 * zoom)
    assert captured["image"].getpixel((mid_x, mid_y)) == (255, 255, 255)
    # ...while a point well away from any text block stays untouched.
    assert captured["image"].getpixel((300, 700)) == (0, 0, 0)


def test_ocr_page_does_not_mask_invisible_text_layer(monkeypatch):
    """The core fix for the invisible-OCR-layer problem: a page whose only
    "native text" is an invisible (render mode 3 / alpha=0) layer overlaid
    on a scanned image — the exact shape of an existing, possibly-bad OCR
    pass — must NOT have that region masked before a fresh OCR pass. Masking
    it would blank out precisely the image content fresh OCR needs to
    re-examine, permanently locking in whatever the old OCR said."""
    doc = fitz.open()
    page = doc.new_page()
    img = Image.new("RGB", (600, 800), color="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(page.rect, stream=buf.getvalue())
    page.insert_text((72, 30), "garbled invisible ocr layer text", render_mode=3)

    captured = {}
    monkeypatch.setattr(
        "ingestion.pdf_parser.pytesseract.image_to_string",
        lambda image, **kwargs: captured.setdefault("image", image) and "irrelevant",
    )

    parser = PDFParser()
    native_text, visible_blocks = parser._extract_visible_native_text(page)
    assert native_text == ""  # invisible layer must not count as native text
    assert visible_blocks == []  # nothing to mask

    parser._ocr_page(page, visible_blocks)
    # Every pixel stays the original black — nothing got masked white,
    # meaning Tesseract received the full, unmodified scanned image.
    assert captured["image"].getpixel((300, 100)) == (0, 0, 0)
    assert captured["image"].getpixel((300, 700)) == (0, 0, 0)


def test_extract_visible_native_text_excludes_invisible_spans():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "invisible ocr text", render_mode=3)
    page.insert_text((72, 100), "visible header text", render_mode=0)

    parser = PDFParser()
    text, blocks = parser._extract_visible_native_text(page)
    assert "invisible ocr text" not in text
    assert "visible header text" in text
    assert len(blocks) == 1


def test_ocr_page_scales_down_zoom_for_oversized_page(monkeypatch):
    parser = PDFParser()
    seen_zoom = {}

    class FakeRect:
        width = 20000  # points — a page far larger than any real-world size
        height = 20000

    class FakePage:
        rect = FakeRect()

        def get_pixmap(self, matrix):
            seen_zoom["x"] = matrix.a  # fitz.Matrix(zoom, zoom) stores zoom in .a/.d
            raise RuntimeError("stop before actually rendering — zoom capture is all this test needs")

    monkeypatch.setattr("ingestion.pdf_parser.fitz.Matrix", lambda x, y: type("M", (), {"a": x, "d": y})())
    try:
        parser._ocr_page(FakePage())
    except Exception:
        pass
    assert seen_zoom["x"] < 2.0  # default zoom scaled down for the oversized page


def test_parse_strips_nul_bytes_from_native_text(tmp_path, monkeypatch):
    """Regression test for a real ingestion failure: a font CMap glyph
    mapped to U+0000 (observed on arXiv's own tex2pdf pipeline, ~40% of a
    real sample affected) survives extraction as a literal NUL byte, which
    Postgres rejects outright once db_save_ingestion tries to store it —
    surfacing as an ingestion failure with no obvious link back to parsing.
    Monkeypatches the extraction step directly rather than trying to coax a
    real font into emitting U+0000 through insert_text()."""
    pdf_path = _make_pdf(tmp_path / "text.pdf", text="placeholder text over fifty characters long for the gate")
    parser = PDFParser()
    # Must clear MIN_NATIVE_TEXT_CHARS (50) on its own, or the length check
    # decides this "needs OCR" and a real Tesseract pass over the actual
    # rendered page (which says "placeholder text...") overwrites it —
    # silently making this test pass or fail on OCR output instead of the
    # NUL-stripping behavior it's meant to isolate.
    injected = "clean text \x00 with an embedded nul byte, long enough to clear the gate on its own"
    assert len(injected) >= PDFParser.MIN_NATIVE_TEXT_CHARS
    monkeypatch.setattr(parser, "_extract_visible_native_text", lambda page: (injected, []))
    result = parser.parse(str(pdf_path))
    assert "\x00" not in result.pages[0]["text"]
    assert result.pages[0]["text"] == injected.replace("\x00", "")


def test_parse_strips_multiple_nul_bytes_from_native_text(tmp_path, monkeypatch):
    pdf_path = _make_pdf(tmp_path / "text.pdf", text="placeholder text over fifty characters long for the gate")
    parser = PDFParser()
    monkeypatch.setattr(
        parser, "_extract_visible_native_text",
        lambda page: ("a\x00b\x00\x00c long enough text to clear the fifty character minimum threshold", []),
    )
    result = parser.parse(str(pdf_path))
    assert "\x00" not in result.pages[0]["text"]
    assert result.pages[0]["text"].startswith("abc")


def test_extract_metadata_strips_nul_bytes(tmp_path):
    pdf_path = _make_pdf(tmp_path / "meta.pdf", text="Enough text content to skip the OCR fallback threshold.")
    doc = fitz.open(str(pdf_path))
    doc.metadata["title"] = "Bad\x00Title"
    doc.metadata["author"] = "Au\x00thor"

    parser = PDFParser()
    meta = parser._extract_metadata(doc, pdf_path)
    doc.close()

    assert meta["title"] == "BadTitle"
    assert meta["author"] == "Author"
    assert "\x00" not in meta["title"] and "\x00" not in meta["author"]


def test_regression_fresh_ocr_corrects_bad_invisible_text_layer(tmp_path):
    """Regression test for the exact risk masking-all-native-text would
    reintroduce: a full-page scan carrying a long (over MIN_NATIVE_TEXT_
    CHARS on its own), deliberately WRONG invisible text layer — simulating
    a bad prior OCR pass baked into the PDF. A fresh parse must:
      1. NOT let that invisible layer's length suppress OCR (it must not
         count as native text at all).
      2. NOT mask the image region under it before OCR (that would blank
         out the very thing a fresh pass needs to re-examine).
      3. Actually recognize the real rendered content, not the bad layer's.

    Uses REAL Tesseract (skipped if unavailable) — this is specifically
    about OCR correctness on this scenario, not just which code path runs."""
    parser = PDFParser()
    if not parser.ocr_available:
        pytest.skip("Tesseract not installed in this environment")

    img = Image.new("RGB", (900, 300), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((30, 30), "THE CORRECT SCANNED CONTENT", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    doc = fitz.open()
    page = doc.new_page(width=900, height=300)
    page.insert_image(page.rect, stream=buf.getvalue())
    # Long, badly garbled invisible OCR layer sitting over the same image —
    # a stand-in for a bad prior OCR pass. Long enough on its own to clear
    # MIN_NATIVE_TEXT_CHARS if it were (wrongly) counted as native text.
    bad_layer = "THE C0RRECT SC4NNED C0NTENT " * 4
    assert len(bad_layer) >= PDFParser.MIN_NATIVE_TEXT_CHARS
    page.insert_text((30, 30), bad_layer, render_mode=3, fontsize=9)
    pdf_path = tmp_path / "bad_ocr_layer.pdf"
    doc.save(str(pdf_path))
    doc.close()

    result = parser.parse(str(pdf_path))
    text = result.pages[0]["text"].upper()
    assert "CORRECT" in text
    assert "SCANNED" in text
    assert "C0RRECT" not in text  # the bad invisible layer's own wording must not survive
    assert result.pages[0]["has_ocr"] is True
