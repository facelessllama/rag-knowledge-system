#!/usr/bin/env python3
"""
One-off migration: old single-unnamed-vector Qdrant collection ->
named dense+sparse hybrid schema (see vector_db/qdrant_client.py).

Chunk text is already stored in Qdrant payload, so this does NOT require
re-parsing original PDFs — it scrolls existing points, recomputes the
sparse (BM25-style) vector from the stored text, reuses the stored dense
vector, and re-upserts into a freshly (re)created collection. It also
backfills the Postgres `documents` table from the same scroll pass —
this is the LAST time a full-collection scroll should ever be necessary;
after this, document metadata lives in Postgres and hybrid search lives
entirely in Qdrant (see api/main.py, rag/retriever.py).

Caveat: `file_hashes` (used for upload dedup) cannot be reconstructed from
Qdrant — the MD5 is computed from the original file bytes, which aren't
stored. If the Postgres `file_hashes` table itself is intact, dedup
history is preserved automatically (this script doesn't touch it). If
Postgres was also wiped, dedup simply starts fresh after migration.

Usage:
    ./backup_qdrant.sh   # snapshot first — this script deletes the collection
    source venv/bin/activate
    python scripts/migrate_to_hybrid_schema.py
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
logger = logging.getLogger("migrate_to_hybrid_schema")

SCROLL_BATCH = 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "knowledge_base"))
    parser.add_argument("--postgres-url", default=os.getenv(
        "POSTGRES_URL", "postgresql://raguser:ragpass@localhost:5432/ragdb"))
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    from lock import run_locked
    run_locked(lambda: _run(args), logger)


def _run(args):
    # This deletes and recreates the whole collection (see below) — a
    # backup_qdrant.sh snapshot taken mid-run could catch an empty or
    # partially-repopulated collection, so this holds the same shared_lock
    # backup's exclusive hold waits out.

    from qdrant_client import QdrantClient
    from vector_db.qdrant_client import DENSE_VECTOR_NAME, VectorStore
    from vector_db.sparse_encoder import build_sparse_vector

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)

    existing = [c.name for c in client.get_collections().collections]
    if args.collection not in existing:
        logger.info(f"Collection '{args.collection}' does not exist — nothing to migrate. "
                    f"A fresh hybrid collection will be created on next app startup.")
        return

    info = client.get_collection(args.collection)
    vectors_config = info.config.params.vectors
    if isinstance(vectors_config, dict) and DENSE_VECTOR_NAME in vectors_config:
        logger.info(f"Collection '{args.collection}' already uses the named hybrid schema — nothing to do.")
        return

    point_count = info.points_count
    logger.info(f"Migrating '{args.collection}' ({point_count} points) to dense+sparse hybrid schema.")
    if not args.yes:
        confirm = input(f"This will DELETE and recreate '{args.collection}'. "
                         f"Make sure you've run ./backup_qdrant.sh first. Continue? [y/N] ")
        if confirm.strip().lower() != "y":
            logger.info("Aborted.")
            return

    logger.info("Scrolling all existing points (with vectors)...")
    all_points = []
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=args.collection, limit=SCROLL_BATCH, offset=offset,
            with_payload=True, with_vectors=True,
        )
        all_points.extend(batch)
        logger.info(f"  scrolled {len(all_points)}/{point_count}")
        if next_offset is None:
            break
        offset = next_offset

    vector_size = len(all_points[0].vector) if all_points and all_points[0].vector else 1024

    logger.info(f"Recreating '{args.collection}' with hybrid schema (vector_size={vector_size})...")
    client.delete_collection(args.collection)
    store = VectorStore(url=args.qdrant_url, collection=args.collection, api_key=args.qdrant_api_key)
    store.create_collection(vector_size=vector_size)

    logger.info("Re-upserting points with sparse vectors + backfilling documents table...")
    from qdrant_client.models import PointStruct

    doc_meta = {}
    new_points = []
    for point in all_points:
        payload = point.payload or {}
        text = payload.get("text", "")
        dense_vector = point.vector if isinstance(point.vector, list) else point.vector.get("", [])
        new_points.append(PointStruct(
            id=point.id,
            vector={DENSE_VECTOR_NAME: dense_vector, "bm25": build_sparse_vector(text)},
            payload=payload,
        ))

        doc_id = payload.get("document_id", "")
        if doc_id:
            if doc_id not in doc_meta:
                doc_meta[doc_id] = {
                    "doc_id": doc_id,
                    "filename": payload.get("filename", "unknown"),
                    "pages": payload.get("pages", 0),
                    "chunks": 0,
                    "size_kb": payload.get("size_kb", 0),
                    "metadata": {},
                    "folder": payload.get("folder", ""),
                    "format": payload.get("format", "pdf"),
                }
            doc_meta[doc_id]["chunks"] += 1

        if len(new_points) >= SCROLL_BATCH:
            client.upsert(collection_name=args.collection, points=new_points)
            new_points = []
    if new_points:
        client.upsert(collection_name=args.collection, points=new_points)

    logger.info(f"Re-upserted {point_count} points with hybrid vectors.")

    logger.info(f"Backfilling {len(doc_meta)} documents into Postgres...")
    import json
    import psycopg2

    conn = psycopg2.connect(args.postgres_url)
    try:
        cur = conn.cursor()
        # doc_id is UUID here to match api/main.py's init_db() — this
        # CREATE TABLE only ever fires on a table that doesn't exist yet
        # (a from-scratch restore), but if it ran first with the old
        # VARCHAR(8), init_db()'s own CREATE TABLE IF NOT EXISTS would then
        # silently no-op against the already-created table, permanently
        # reintroducing the 32-bit-doc_id collision risk (see api/main.py's
        # documents table comment) on any fresh restore that happens to run
        # this script before the app's first startup.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id UUID PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                pages INTEGER DEFAULT 0,
                chunks INTEGER DEFAULT 0,
                size_kb REAL DEFAULT 0,
                metadata JSONB DEFAULT '{}',
                folder VARCHAR(255) DEFAULT '',
                format VARCHAR(10) DEFAULT 'pdf',
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        for doc in doc_meta.values():
            cur.execute(
                """
                INSERT INTO documents (doc_id, filename, pages, chunks, size_kb, metadata, folder, format)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE SET
                    filename = EXCLUDED.filename, pages = EXCLUDED.pages, chunks = EXCLUDED.chunks,
                    size_kb = EXCLUDED.size_kb, folder = EXCLUDED.folder, format = EXCLUDED.format
                """,
                (doc["doc_id"], doc["filename"], doc["pages"], doc["chunks"], doc["size_kb"],
                 json.dumps(doc["metadata"]), doc["folder"], doc["format"]),
            )
        conn.commit()
    finally:
        conn.close()

    logger.info("Migration complete.")


if __name__ == "__main__":
    main()
