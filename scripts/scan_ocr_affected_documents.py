#!/usr/bin/env python3
"""
Cheap, READ-ONLY scan for documents likely affected by the OCR-trigger fix
in ingestion/pdf_parser.py. Previously, OCR only ran when a page's native
text was under 50 chars, regardless of how much of the page was covered by
a placed image — a scanned body sitting under a long native header/footer/
watermark was never even considered for OCR, silently dropping its entire
content. The fix adds an image-coverage trigger alongside the length one.

Flags a page as "likely affected" when the NEW image-coverage trigger
would fire (PDFParser._image_coverage_ratio() >= IMAGE_COVERAGE_OCR_
THRESHOLD) but the OLD length-only trigger would NOT have (native text
length >= MIN_NATIVE_TEXT_CHARS) — exactly the scenario the old heuristic
missed. A page already under the old length threshold was already an OCR
candidate before this fix, so it's not "newly affected" by it.

Deliberately does NOT re-run OCR, re-chunk, or re-embed anything — this
opens each PDF and reads native text + image placement per page only,
nothing else, so it's fast enough to run across a large corpus as a
pre-filter. The output feeds directly into
scripts/backfill_document_pages.py --doc-ids, which does the expensive
part (re-parse WITH OCR + chunk + embed + Qdrant compare) only for the
subset this scan actually flags, instead of the whole corpus.

This does NOT catch every case the fix addresses — a page whose native
text was already short (the length trigger alone already covered it) but
whose OCR failed or was skipped for some other reason (Tesseract
unavailable at ingestion time, budget exhausted, OCR returned empty and
the old code blanked out the native text) is invisible to this scan, since
that scenario doesn't depend on image coverage. Those pages are still
caught by a full (non---doc-ids) backfill_document_pages.py run, which
compares actual extracted TEXT, not just this scan's heuristic — running
this scan first is an optional speed-up for a large corpus, not a
substitute for eventually running the full backfill.

Usage:
    source venv/bin/activate
    python scripts/scan_ocr_affected_documents.py
    python scripts/scan_ocr_affected_documents.py --json > affected.json
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz
import psycopg2
from dotenv import load_dotenv

from ingestion.pdf_parser import PDFParser

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scan_ocr_affected_documents")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON instead of a log summary")
    args = ap.parse_args()

    postgres_url = os.getenv("POSTGRES_URL", "postgresql://raguser:ragpass@localhost:5432/ragdb")
    upload_dir = Path("uploads")

    conn = psycopg2.connect(postgres_url)
    cur = conn.cursor()
    cur.execute("SELECT doc_id, filename FROM documents WHERE format = 'pdf'")
    docs = cur.fetchall()

    # _image_coverage_ratio() is pure page geometry — no Tesseract call, so
    # instantiating without checking OCR availability is fine here.
    parser = PDFParser()

    affected = []
    missing_files = []
    open_failures = []
    total_pages_scanned = 0
    total_pages_affected = 0

    for doc_id, filename in docs:
        file_path = next(upload_dir.glob(f"{doc_id}_*"), None)
        if not file_path:
            missing_files.append({"doc_id": str(doc_id), "filename": filename})
            continue
        try:
            doc = fitz.open(str(file_path))
        except Exception as e:
            open_failures.append({"doc_id": str(doc_id), "filename": filename, "error": str(e)})
            continue

        affected_pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            total_pages_scanned += 1
            native_text = page.get_text().strip()
            if len(native_text) < PDFParser.MIN_NATIVE_TEXT_CHARS:
                continue  # old heuristic already would have tried OCR here — not newly affected
            if parser._image_coverage_ratio(page) >= PDFParser.IMAGE_COVERAGE_OCR_THRESHOLD:
                affected_pages.append(page_num + 1)
        doc.close()

        if affected_pages:
            affected.append({"doc_id": str(doc_id), "filename": filename, "pages": affected_pages})
            total_pages_affected += len(affected_pages)

    result = {
        "documents_scanned": len(docs),
        "documents_affected": len(affected),
        "pages_scanned": total_pages_scanned,
        "pages_affected": total_pages_affected,
        "missing_files": missing_files,
        "open_failures": open_failures,
        "affected": affected,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    logger.info(f"Scanned {result['documents_scanned']} PDF documents ({result['pages_scanned']} pages)")
    logger.info(f"Likely affected by the OCR-trigger fix: {result['documents_affected']} documents, {result['pages_affected']} pages")
    if missing_files:
        logger.warning(f"{len(missing_files)} document(s) had no file on disk to scan")
    if open_failures:
        logger.warning(f"{len(open_failures)} document(s) failed to open: {open_failures}")
    for a in affected:
        logger.info(f"  {a['doc_id']} {a['filename']} — pages {a['pages']}")
    if affected:
        doc_ids = ",".join(a["doc_id"] for a in affected)
        logger.info("Targeted backfill (dry run first):")
        logger.info(f"  python scripts/backfill_document_pages.py --dry-run --doc-ids {doc_ids}")
    else:
        logger.info("Nothing flagged — a full backfill_document_pages.py run is still worth doing at "
                     "least once (see its own docstring: it catches text differences this scan can't, "
                     "like OCR having been unavailable/budget-exhausted at original ingestion time).")


if __name__ == "__main__":
    main()
