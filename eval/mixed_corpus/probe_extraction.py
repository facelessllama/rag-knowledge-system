#!/usr/bin/env python3
"""Pre-flight extraction/chunking probe for the mixed corpus.

Runs the SAME production classes the live API uses on upload
(ingestion.pdf_parser.PDFParser + ingestion.chunker.SmartChunker), against
every document in manifest.json, WITHOUT touching Qdrant/Postgres/embeddings.
The point is to know, before spending time on a live ingestion run, which
documents will predictably fail the API's own "Could not extract text from
document" check (api/main.py: `if not chunks: raise ValueError(...)`) — most
expected in hard/old_scans and hard/old_scans_math, whose olmOCR-bench source
pages are handwritten/formula-dense scans that plain Tesseract (no math/
handwriting model) is not equipped to read.

Writes eval/mixed_corpus/extraction_probe.json (one row per manifest entry)
and patches predicted_chunks/predicted_status onto manifest.json itself, so
later golden-question building can trivially filter to documents that will
actually make it into the index.

Usage:
    source venv/bin/activate
    python eval/mixed_corpus/probe_extraction.py
"""
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from ingestion.pdf_parser import PDFParser
from ingestion.chunker import SmartChunker

CORPUS_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
PROBE_OUT_PATH = CORPUS_DIR / "extraction_probe.json"

# Mirrors the exact settings api/main.py constructs PDFParser/SmartChunker
# with, read from the same .env — a probe run under different settings would
# predict a different outcome than the real ingestion run will produce.
OCR_LANGUAGE = os.getenv("PDF_OCR_LANGUAGE", "eng")
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "500"))
OCR_TIMEOUT_SECONDS = int(os.getenv("OCR_TIMEOUT_SECONDS", "60"))
MAX_TOTAL_OCR_SECONDS = int(os.getenv("MAX_TOTAL_OCR_SECONDS", "600"))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Passing the chunker's min_chunk_size gate (>=100 chars) only means a page
# produced *some* text — it says nothing about whether OCR actually read the
# page correctly. A badly garbled scan (illustration, handwriting, dense
# formula notation) routinely clears 100 chars of pure noise ("| a ariel\nes
# ae i ee AC gn") and gets chunked/embedded/indexed anyway, quietly
# polluting retrieval for unrelated documents. Real English prose is
# roughly 30-50% function/stop words; character-soup OCR output is not —
# collisions are rare because this only counts tokens of length >= 3, which
# excludes single-letter math variables ("a", "b", "x") that would otherwise
# look like real stopword hits in formula-heavy pages.
_STOPWORDS = frozenset("""
the and of to in a is that for on with as by at from this be are or an it
was were not have has had but which their its into can may will would
should could been being also than then there these those such other some
any all each more most one two three used using use based between within
however therefore thus while during over under about after before above
below out off up down again further once here where when why how what who
whom whose all both each few more most other some such only own same so
than too very just don now first second third table figure section chapter
""".split())


# Catches the corpus's actual non-English contamination (found by hand
# after this heuristic flagged it as "garbage": FAA safety brochures have
# Spanish/Portuguese translations sitting right next to the English
# original on the same page — "faa_Span_Fatigue.pdf" is Spanish, not an
# abbreviation of anything English). A low English-stopword ratio alone
# can't tell "not English" apart from "genuine OCR noise"; this second,
# small Spanish/Portuguese marker-word set resolves the ambiguity instead
# of silently lumping both under one label a human would have to
# re-diagnose by hand anyway.
_NON_ENGLISH_HINTS = frozenset("de la el que para los una es del por con las como su al".split())


def _stopword_ratio(text: str) -> tuple[float, int, float]:
    tokens = re.findall(r"[A-Za-z]{3,}", text.lower())
    if not tokens:
        return 0.0, 0, 0.0
    hits = sum(1 for t in tokens if t in _STOPWORDS)
    non_english_hits = sum(1 for t in tokens if t in _NON_ENGLISH_HINTS)
    return hits / len(tokens), len(tokens), non_english_hits / len(tokens)


def probe_one(parser: PDFParser, chunker: SmartChunker, path: Path) -> dict:
    t0 = time.time()
    try:
        parsed = parser.parse(str(path))
    except Exception as exc:
        return {
            "predicted_status": "would_error",
            "error": str(exc)[:300],
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    total_chars = sum(p["char_count"] for p in parsed.pages)
    ocr_pages = sum(1 for p in parsed.pages if p["has_ocr"])
    empty_pages = sum(1 for p in parsed.pages if not p["text"])

    chunks = chunker.chunk_document(parsed.pages, doc_id="probe")

    full_text = "\n".join(p["text"] for p in parsed.pages)
    stopword_ratio, token_count, non_english_ratio = _stopword_ratio(full_text)
    if not chunks:
        quality_flag = "n/a"  # already rejected by the chunker gate
    elif token_count < 15:
        quality_flag = "too_short_to_judge"
    elif stopword_ratio < 0.12:
        # Both look "low English-stopword" from the outside; only the
        # Spanish/Portuguese marker-word ratio distinguishes which one it
        # actually is. Threshold well below the English one on purpose —
        # this only needs to catch a clear signal, not classify precisely.
        quality_flag = "likely_non_english" if non_english_ratio > 0.04 else "likely_garbage"
    else:
        quality_flag = "likely_ok"

    return {
        "predicted_status": "would_ingest" if chunks else "would_reject_no_text",
        "total_pages": parsed.total_pages,
        "total_chars": total_chars,
        "avg_chars_per_page": round(total_chars / parsed.total_pages, 1) if parsed.total_pages else 0,
        "ocr_pages": ocr_pages,
        "empty_pages": empty_pages,
        "predicted_chunks": len(chunks),
        "stopword_ratio": round(stopword_ratio, 3),
        "token_count": token_count,
        "quality_flag": quality_flag,
        "elapsed_ms": int((time.time() - t0) * 1000),
    }


def main():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    parser = PDFParser(
        ocr_language=OCR_LANGUAGE, max_pages=MAX_PDF_PAGES,
        ocr_timeout_seconds=OCR_TIMEOUT_SECONDS, max_total_ocr_seconds=MAX_TOTAL_OCR_SECONDS,
    )
    chunker = SmartChunker(chunk_size=MAX_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    results = []
    for i, entry in enumerate(manifest, 1):
        path = CORPUS_DIR / entry["relative_path"]
        row = {"relative_path": entry["relative_path"], "category": entry["category"],
               "source_id": entry["source_id"], "split": entry["split"]}
        row.update(probe_one(parser, chunker, path))
        results.append(row)
        if i % 25 == 0 or i == len(manifest):
            print(f"[{i}/{len(manifest)}] {row['predicted_status']:22s} {entry['category']:14s} {entry['source_id'][:50]}", flush=True)

    PROBE_OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Patch predicted_status/predicted_chunks back onto manifest.json so
    # downstream dataset-building code can filter without re-running the probe.
    by_path = {r["relative_path"]: r for r in results}
    for entry in manifest:
        r = by_path.get(entry["relative_path"])
        if r:
            entry["predicted_status"] = r["predicted_status"]
            entry["predicted_chunks"] = r.get("predicted_chunks")
            entry["quality_flag"] = r.get("quality_flag")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ── summary ──
    print(f"\n{'='*70}")
    print("EXTRACTION PROBE SUMMARY (predicted outcome, same code path as /upload)")
    print(f"{'='*70}")
    by_cat: dict[str, dict[str, int]] = {}
    for r in results:
        cat = r["category"]
        status = r["predicted_status"]
        by_cat.setdefault(cat, {}).setdefault(status, 0)
        by_cat[cat][status] += 1
        qf = r.get("quality_flag")
        if qf in ("likely_garbage", "likely_non_english"):
            by_cat[cat][qf] = by_cat[cat].get(qf, 0) + 1

    for cat in sorted(by_cat):
        counts = by_cat[cat]
        total = sum(v for k, v in counts.items() if k not in ("likely_garbage", "likely_non_english"))
        reject = counts.get("would_reject_no_text", 0) + counts.get("would_error", 0)
        garbage = counts.get("likely_garbage", 0)
        non_eng = counts.get("likely_non_english", 0)
        flag = "  <-- expect ingestion failures" if reject else ""
        flag += "  <-- OCR noise" if garbage else ""
        flag += "  <-- NON-ENGLISH, exclude from corpus" if non_eng else ""
        print(f"{cat:16s} total={total:4d}  ingest_ok={counts.get('would_ingest', 0):4d}  "
              f"reject_no_text={counts.get('would_reject_no_text', 0):4d}  "
              f"error={counts.get('would_error', 0):4d}  likely_garbage={garbage:4d}  "
              f"non_english={non_eng:4d}{flag}")

    total_reject = sum(1 for r in results if r["predicted_status"] != "would_ingest")
    total_garbage = sum(1 for r in results if r.get("quality_flag") == "likely_garbage")
    total_non_eng = sum(1 for r in results if r.get("quality_flag") == "likely_non_english")
    print(f"\nTotal: {len(results)} documents")
    print(f"  {total_reject} predicted NOT to make it into the index ({100*total_reject/len(results):.1f}%)")
    print(f"  {total_garbage} more predicted to ingest but read as OCR noise, not real text "
          f"({100*total_garbage/len(results):.1f}%) — heuristic triage signal, spot-check before trusting")
    print(f"  {total_non_eng} more flagged non-English (Spanish/Portuguese marker words, not English "
          f"stopwords) — this corpus is meant to be English-only, these should be REMOVED, not just excluded "
          f"from questions")
    excl = total_reject + total_garbage + total_non_eng
    print(f"  => {excl} of {len(results)} ({100*excl/len(results):.1f}%) should not feed fact-based golden questions")
    print(f"Per-document detail: {PROBE_OUT_PATH}")


if __name__ == "__main__":
    main()
