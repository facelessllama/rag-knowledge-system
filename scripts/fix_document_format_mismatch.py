#!/usr/bin/env python3
"""
Corrects `documents.format` for rows where it doesn't match the actual file
extension on disk — found live: 200 of 400 documents (the entire "UK
Regulations" TXT corpus) carry `format='pdf'` in Postgres despite being real
`.txt` files. get_document_page() (api/documents.py) branches on this field
to decide whether to read the page live off disk (TXT) or from the
`document_pages` table (PDF, populated only at PDF ingestion — see
scripts/backfill_document_pages.py). A TXT document mis-tagged 'pdf' has no
`document_pages` row, so every click on its source citation 404s with "Page
text not indexed for this document" — surfaced in the UI as "Failed to load
document."

Root cause: `scripts/migrate_to_hybrid_schema.py`'s one-time Qdrant-schema
migration backfilled Postgres `documents` from old chunk payloads via
`payload.get("format", "pdf")` — payloads from before `format` was tracked
per-chunk had no such key, so every one of them silently defaulted to
'pdf' regardless of the source file's real extension. The live `/upload`
path (api/upload.py's `pick_parser()`) has never had this bug — it derives
format from the filename extension directly — so this is purely residue
from that one historical migration, not something that can recur through
ordinary use.

The actual file content and Qdrant chunks are unaffected — retrieval and
generation already work correctly for these documents (`format` is not
stored on any chunk payload, only on the `documents` row — confirmed by
grep, not assumed). This script fixes ONLY the metadata field driving the
document-viewer's TXT/PDF branch.

Rule: format is derived from the filename extension (.txt -> 'txt',
.pdf -> 'pdf'), the same mapping pick_parser() uses for new uploads — not
hardcoded to "always txt" — so this also catches the mirror-image mistake
(a real .pdf mistagged 'txt') if one ever turns up, though a live check
found none.

Usage:
    source venv/bin/activate
    python scripts/fix_document_format_mismatch.py --dry-run   # report only, no writes
    python scripts/fix_document_format_mismatch.py              # apply

Requires an app restart afterward (or waiting for the next one) — like
scripts/backfill_document_pages.py, this writes Postgres directly; the
running process's in-memory `documents_registry` only reloads from
Postgres at startup (see api/db.py's init_db()).
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fix_document_format_mismatch")

# Same extension -> format mapping api/upload.py's pick_parser() uses for
# new uploads (see PARSERS_BY_EXT, built in api/main.py's startup()) —
# kept as a small local literal here rather than importing PARSERS_BY_EXT
# itself, since that dict holds live parser *instances* built at server
# startup, not a plain mapping this standalone script could reuse directly.
EXT_TO_FORMAT = {"pdf": "pdf", "txt": "txt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    from lock import run_locked
    run_locked(lambda: _run(args), logger)


def _run(args):
    import psycopg2

    postgres_url = os.getenv("POSTGRES_URL", "postgresql://raguser:ragpass@localhost:5432/ragdb")
    conn = psycopg2.connect(postgres_url)
    cur = conn.cursor()
    cur.execute("SELECT doc_id, filename, format FROM documents")
    rows = cur.fetchall()
    logger.info(f"{len(rows)} documents in Postgres to check")

    mismatched = []
    unknown_ext = []
    for doc_id, filename, doc_format in rows:
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        expected = EXT_TO_FORMAT.get(ext)
        if expected is None:
            unknown_ext.append((doc_id, filename))
            continue
        if doc_format != expected:
            mismatched.append((doc_id, filename, doc_format, expected))

    logger.info(f"{'[DRY RUN] ' if args.dry_run else ''}{len(mismatched)} mismatched, "
                f"{len(unknown_ext)} unrecognized extension (left untouched)")

    for doc_id, filename, old_format, new_format in mismatched:
        logger.info(f"  {'WOULD FIX' if args.dry_run else 'FIXING'}  "
                    f"{doc_id} {filename}: format {old_format!r} -> {new_format!r}")
        if not args.dry_run:
            cur.execute("UPDATE documents SET format = %s WHERE doc_id = %s", (new_format, doc_id))

    if not args.dry_run and mismatched:
        conn.commit()
        logger.info(f"Committed {len(mismatched)} fixes. Restart the app to pick them up in "
                    f"the live documents_registry (see api/db.py's init_db()).")

    for doc_id, filename in unknown_ext:
        logger.warning(f"  UNRECOGNIZED EXTENSION (skipped) {doc_id} {filename}")


if __name__ == "__main__":
    main()
