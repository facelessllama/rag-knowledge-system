#!/usr/bin/env python3
"""
Self-verifying backfill for document_pages (see api/main.py's document_pages
table / db_save_ingestion(), and the /documents/{doc_id}/pages/{page_num}
endpoint that superseded /pdf/{doc_id}/highlights). Populates per-page
normalized text for PDFs ingested before that table existed.

PDF only — TXT documents are read live off disk at request time (see
get_document_page() in api/main.py): TxtParser + normalize_whitespace() is
a pure function with no external library version to drift, so there is
nothing for this backfill to protect there, unlike PDF's PyMuPDF/Tesseract
extraction.

Does NOT trust that re-parsing a document today reproduces byte-identical
output to what was originally chunked — a PyMuPDF or Tesseract version bump
(or config change) between then and now would otherwise silently desync
freshly-persisted page text from the char_start/char_end offsets already
live in Qdrant, which is exactly the bug this whole migration exists to fix,
just moved into the backfill instead of the highlight path.

For every PDF document in Postgres (documents table):
  1. Re-parse the file on disk with the CURRENT parser/chunker (same config
     api/main.py's /upload uses — MAX_CHUNK_SIZE, CHUNK_OVERLAP, PDF_OCR_
     LANGUAGE, EMBEDDING_MODEL env vars).
  2. Scroll ALL of its existing points out of Qdrant, keeping every point
     ID per chunk_id (not just the last one seen) — two live points sharing
     a chunk_id is itself a sign of a botched prior write and must fail
     verification, not be silently collapsed into "one merged chunk". A
     point matching this document_id but missing chunk_id entirely
     (malformed/corrupt write) is tracked separately as an invalid point —
     it used to just be skipped over, invisible to comparison and never
     collected for deletion, so it would survive every backfill forever.
  3. Compare, chunk_id for chunk_id: text, page_num, chunk_index,
     char_start, char_end, has_ocr. Any missing chunk_id, extra chunk_id,
     duplicate chunk_id, invalid (chunk_id-less) point, or field mismatch
     fails verification.
  4. Independently re-verify the invariant
     normalized_page_text[char_start:char_end] == chunk.text for every
     chunk — belt-and-braces on top of #3.
  5. Only if everything matches: persist page text as-is. Otherwise the
     document is out of sync with what's actually indexed:
       a. chunk + embed the fresh chunks and upsert them (new random
          uuid4() point IDs — see vector_db/qdrant_client.py::
          upsert_chunks — so this never collides with the old points).
       b. only once the new points are safely in Qdrant, delete the OLD
          point IDs collected in step 2 (by exact point ID, not a
          document_id filter — a filter would also delete the new points
          we just added, since they share the same document_id).
       c. refresh documents.chunks/pages, THEN persist page text.
     Embedding a fresh set before deleting the old one means a mid-way
     failure (GPU error, Qdrant write error) leaves the document with its
     OLD (stale but present) points intact, never with zero points for the
     document — a temporary duplicate is far cheaper than a hole.

Pages with zero chunks in Qdrant (shorter than SmartChunker.min_chunk_size,
or a dropped trailing remainder) can't be verified against an existing
chunk — reported as a separate "unverifiable" count, not silently folded
into "verified". This is not a correctness gap: no existing /query source
has offsets pointing into such a page, so there's nothing for the viewer to
get wrong there.

Usage:
    source venv/bin/activate
    python scripts/backfill_document_pages.py --dry-run   # report only, no writes
    python scripts/backfill_document_pages.py              # apply
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_document_pages")

FIELDS_TO_COMPARE = ["text", "page_num", "chunk_index", "char_start", "char_end", "has_ocr"]


def scroll_all_points(client, collection, doc_id):
    """Returns ({chunk_id: [{"point_id":..., "payload":...}, ...]}, invalid_point_ids)
    — a list per chunk_id, not a single overwritten entry, so duplicate live
    points sharing a chunk_id are visible to the caller instead of silently
    collapsing to whichever one was scrolled last. A point matching this
    document_id but missing chunk_id entirely (corrupt/malformed write) used
    to just be skipped — invisible to compare_chunks, and never collected
    for deletion on reindex, so it would survive forever. invalid_point_ids
    carries those out separately so the caller can force a mismatch and
    delete them along with everything else."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    points = {}
    invalid_point_ids = []
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]),
            limit=500, offset=offset, with_payload=True, with_vectors=False,
        )
        for p in batch:
            payload = p.payload or {}
            cid = payload.get("chunk_id")
            if cid:
                points.setdefault(cid, []).append({"point_id": p.id, "payload": payload})
            else:
                invalid_point_ids.append(p.id)
        if next_offset is None:
            break
        offset = next_offset
    return points, invalid_point_ids


def compare_chunks(old_points, invalid_point_ids, new_chunks):
    old_ids = set(old_points.keys())
    new_by_id = {c.chunk_id: c for c in new_chunks}
    new_ids = set(new_by_id.keys())

    missing = old_ids - new_ids  # existed in Qdrant before, chunker no longer produces it
    extra = new_ids - old_ids    # chunker produces it now, wasn't in Qdrant before
    duplicates = {cid: [e["point_id"] for e in entries] for cid, entries in old_points.items() if len(entries) > 1}
    mismatched = []

    for cid in old_ids & new_ids:
        entries = old_points[cid]
        old = entries[0]["payload"]
        new = new_by_id[cid]
        for field in FIELDS_TO_COMPARE:
            old_val = old.get(field)
            new_val = getattr(new, field)
            if field == "has_ocr":
                old_val, new_val = bool(old_val), bool(new_val)
            if old_val != new_val:
                mismatched.append((cid, field, old_val, new_val))

    ok = not missing and not extra and not mismatched and not duplicates and not invalid_point_ids
    return ok, {"missing": missing, "extra": extra, "mismatched": mismatched, "duplicates": duplicates,
                "invalid_point_ids": invalid_point_ids}


def verify_offset_invariant(chunks, normalized_pages):
    failures = []
    for c in chunks:
        page_text = normalized_pages.get(c.page_num, "")
        if c.char_start is None or c.char_end is None:
            failures.append((c.chunk_id, "no offsets"))
            continue
        if page_text[c.char_start:c.char_end] != c.text:
            failures.append((c.chunk_id, "offset mismatch"))
    return failures


def save_pages(conn, doc_id, pages, normalize_whitespace):
    cur = conn.cursor()
    for pg in pages:
        norm = normalize_whitespace(pg["text"])
        cur.execute(
            """
            INSERT INTO document_pages (document_id, page_num, normalized_text, has_ocr, char_count)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (document_id, page_num) DO UPDATE SET
                normalized_text = EXCLUDED.normalized_text,
                has_ocr = EXCLUDED.has_ocr,
                char_count = EXCLUDED.char_count
            """,
            (doc_id, pg["page_num"], norm, pg.get("has_ocr", False), len(norm)),
        )
    conn.commit()


def save_document(conn, doc_id, filename, pages, chunks, size_kb, metadata, folder, doc_format):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO documents (doc_id, filename, pages, chunks, size_kb, metadata, folder, format)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (doc_id) DO UPDATE SET
            filename = EXCLUDED.filename, pages = EXCLUDED.pages, chunks = EXCLUDED.chunks,
            size_kb = EXCLUDED.size_kb, metadata = EXCLUDED.metadata, folder = EXCLUDED.folder,
            format = EXCLUDED.format
        """,
        (doc_id, filename, pages, chunks, size_kb, json.dumps(metadata or {}), folder or "", doc_format),
    )
    conn.commit()


def reindex_document(vector_store, embedder, chunk_context_text, doc_id, filename, folder,
                      parsed, chunks, old_point_ids):
    """Embed + upsert the fresh chunk set FIRST (new random uuid4() point
    IDs — never collides with the old ones), and only delete the old point
    IDs — collected by the caller from scroll_all_points(), never derived
    by re-filtering on document_id — once the new points are confirmed
    written. A crash between upsert and delete leaves old+new both present
    (a temporary duplicate the next verification pass will catch and
    re-reindex), never zero points for the document."""
    for c in chunks:
        c.filename = filename
        c.pages = parsed.total_pages
        c.folder = folder or ""
    texts = [chunk_context_text(c) for c in chunks]
    vectors = embedder.embed_batch(texts)
    vector_store.upsert_chunks(chunks, vectors)

    if old_point_ids:
        vector_store.client.delete(
            collection_name=vector_store.collection,
            points_selector=old_point_ids,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    from lock import run_locked
    run_locked(lambda: _run(args), logger)


def _run(args):
    import psycopg2

    from ingestion.chunker import SmartChunker, normalize_whitespace, chunk_context_text
    from ingestion.pdf_parser import PDFParser
    from vector_db.qdrant_client import VectorStore

    postgres_url = os.getenv("POSTGRES_URL", "postgresql://raguser:ragpass@localhost:5432/ragdb")
    upload_dir = Path("uploads")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = os.getenv("QDRANT_COLLECTION", "knowledge_base")

    conn = psycopg2.connect(postgres_url)
    cur = conn.cursor()
    cur.execute("SELECT doc_id, filename, pages, chunks, size_kb, metadata, folder, format "
                "FROM documents WHERE format = 'pdf'")
    docs = cur.fetchall()
    logger.info(f"{len(docs)} PDF documents in Postgres to check (TXT is read live off disk — see api/main.py)")

    pdf_parser = PDFParser(ocr_language=os.getenv("PDF_OCR_LANGUAGE", "eng"))
    chunker = SmartChunker(
        chunk_size=int(os.getenv("MAX_CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
    )
    vector_store = VectorStore(url=qdrant_url, collection=collection, api_key=os.getenv("QDRANT_API_KEY"))

    embedder = None  # lazy — only loaded onto the GPU if a reindex actually turns out to be needed

    verified, reindexed = 0, 0
    failed = []
    unverifiable_pages_total = 0
    mismatch_details = []

    for doc_id, filename, pages_count, chunks_count, size_kb, metadata, folder, doc_format in docs:
        file_path = next(upload_dir.glob(f"{doc_id}_*"), None)
        if not file_path:
            failed.append((doc_id, filename, "file not found on disk"))
            continue

        try:
            parsed = pdf_parser.parse(str(file_path))
        except Exception as e:
            failed.append((doc_id, filename, f"parse failed: {e}"))
            continue

        chunks = chunker.chunk_document(parsed.pages, doc_id)
        normalized_pages = {pg["page_num"]: normalize_whitespace(pg["text"]) for pg in parsed.pages}

        old_points, invalid_point_ids = scroll_all_points(vector_store.client, collection, doc_id)
        ok, report = compare_chunks(old_points, invalid_point_ids, chunks)

        pages_with_chunks = {c.page_num for c in chunks}
        unverifiable = [pg["page_num"] for pg in parsed.pages if pg["page_num"] not in pages_with_chunks]
        unverifiable_pages_total += len(unverifiable)

        if ok:
            offset_failures = verify_offset_invariant(chunks, normalized_pages)
            if offset_failures:
                failed.append((doc_id, filename, f"offset invariant failed: {offset_failures[:3]}"))
                continue
            logger.info(f"VERIFIED   {doc_id} {filename} "
                        f"({len(chunks)} chunks, {len(unverifiable)} unverifiable pages)")
            verified += 1
            if not args.dry_run:
                save_pages(conn, doc_id, parsed.pages, normalize_whitespace)
        else:
            logger.warning(f"MISMATCH   {doc_id} {filename} — "
                            f"missing={len(report['missing'])} extra={len(report['extra'])} "
                            f"field_mismatches={len(report['mismatched'])} "
                            f"duplicate_chunk_ids={len(report['duplicates'])} "
                            f"invalid_points={len(report['invalid_point_ids'])} -> reindexing")
            mismatch_details.append((doc_id, filename, report))
            reindexed += 1
            if not args.dry_run:
                if embedder is None:
                    from embeddings.embedding_service import EmbeddingService
                    embedder = EmbeddingService(model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
                # Every live point for this document_id must go — including
                # chunk_id-less ones (invalid_point_ids) — since the fresh
                # set fully replaces it.
                old_point_ids = [e["point_id"] for entries in old_points.values() for e in entries]
                old_point_ids += invalid_point_ids
                reindex_document(vector_store, embedder, chunk_context_text,
                                  doc_id, filename, folder, parsed, chunks, old_point_ids)
                save_document(conn, doc_id, filename, parsed.total_pages, len(chunks),
                              size_kb, metadata, folder, doc_format)
                save_pages(conn, doc_id, parsed.pages, normalize_whitespace)

    logger.info("=" * 70)
    logger.info(f"{'[DRY RUN] ' if args.dry_run else ''}Done.")
    logger.info(f"  verified & persisted            : {verified}")
    logger.info(f"  mismatched -> reindexed          : {reindexed}")
    logger.info(f"  failed (see below)               : {len(failed)}")
    logger.info(f"  pages with no verifiable chunk    : {unverifiable_pages_total} "
                f"(short/dropped pages — no existing source references these, not a correctness gap)")
    for doc_id, filename, reason in failed:
        logger.error(f"  FAILED {doc_id} {filename}: {reason}")
    for doc_id, filename, report in mismatch_details:
        logger.info(f"  MISMATCH DETAIL {doc_id} {filename}: "
                    f"missing={list(report['missing'])[:5]} extra={list(report['extra'])[:5]} "
                    f"duplicates={dict(list(report['duplicates'].items())[:3])} "
                    f"invalid_point_ids={report['invalid_point_ids'][:5]} "
                    f"mismatched_sample={report['mismatched'][:3]}")


if __name__ == "__main__":
    main()
