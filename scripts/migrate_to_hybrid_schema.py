#!/usr/bin/env python3
"""
One-off migration: old single-unnamed-vector Qdrant collection ->
named dense+sparse hybrid schema (see vector_db/qdrant_client.py).

Chunk text is already stored in Qdrant payload, so this does NOT require
re-parsing original PDFs — it scrolls existing points, recomputes the
sparse (BM25-style) vector from the stored text, reuses the stored dense
vector, and copies into a freshly created SEPARATE physical collection.
It also backfills the Postgres `documents` table from the same scroll
pass — this is the LAST time a full-collection scroll should ever be
necessary; after this, document metadata lives in Postgres and hybrid
search lives entirely in Qdrant (see api/main.py, rag/retriever.py).

Zero-downtime-ish cutover via a Qdrant alias, not delete+rebuild-in-place:
  1. Copy every point from the OLD collection into a NEW, separate physical
     collection (`<old>_hybrid_<unix-ts>`) with the hybrid schema — one
     scroll batch at a time, upserted immediately, never holding the whole
     corpus in memory.
  2. Verify the NEW collection before touching the OLD one at all: point
     count matches, schema is the expected named dense+sparse hybrid, and a
     sample (or, with --full-verify, every) point's payload matches byte-
     for-byte old vs new.
  3. Only if verification passes: atomically point the collection name the
     app actually uses at the NEW physical collection (a Qdrant alias), and
     get rid of the OLD physical collection's claim on that name.
  4. Backfill Postgres `documents` from the same pass.

The OLD collection/data is never deleted until step 3, and only after step
2's verification has already passed — a failure anywhere in steps 1-2
leaves the OLD collection completely untouched and the NEW one sitting
around for inspection, never a state with fewer live points than before.

Locking: this holds lock.exclusive_lock() (see its docstring) for the ENTIRE
copy+verify+cutover, not just the cutover — a shared_lock() alone (what the
old version of this script used) does NOT stop concurrent API uploads/
deletes from running against the OLD collection while a scroll cursor is
mid-flight; a write landing in an offset region the cursor already passed
would silently never make it into the new collection. This blocks new
uploads/deletes/folder changes for the operation's whole duration — expected
to be fast regardless, since no re-embedding happens (the stored dense
vector is reused as-is, only the lightweight BM25 sparse vector is
recomputed from stored text).

Caveat: `file_hashes` (used for upload dedup) cannot be reconstructed from
Qdrant — the MD5 is computed from the original file bytes, which aren't
stored. If the Postgres `file_hashes` table itself is intact, dedup
history is preserved automatically (this script doesn't touch it). If
Postgres was also wiped, dedup simply starts fresh after migration.

Usage:
    ./backup_qdrant.sh   # snapshot first, still recommended even though
                         # the old collection is never touched until after
                         # verification passes
    source venv/bin/activate
    python scripts/migrate_to_hybrid_schema.py
    python scripts/migrate_to_hybrid_schema.py --full-verify   # exhaustive payload check instead of a sample
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("migrate_to_hybrid_schema")

SCROLL_BATCH = 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "knowledge_base"))
    parser.add_argument("--postgres-url", default=os.getenv(
        "POSTGRES_URL", "postgresql://raguser:ragpass@localhost:5432/ragdb"))
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--verify-sample", type=int, default=200,
                         help="Number of points to spot-check (exact payload equality, old vs new) "
                              "after copying. Ignored if --full-verify is set. (default: 200)")
    parser.add_argument("--full-verify", action="store_true",
                         help="Check EVERY point's payload old vs new instead of a sample — slower, "
                              "but exhaustive.")
    parser.add_argument("--exclusive-lock-wait-seconds", type=float, default=60.0,
                         help="How long to wait for in-flight API requests to finish before acquiring "
                              "the exclusive maintenance lock (default: 60s)")
    args = parser.parse_args()
    _run(args)


def _run(args):
    from qdrant_client import QdrantClient
    from vector_db.qdrant_client import DENSE_VECTOR_NAME

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)

    try:
        info = client.get_collection(args.collection)
    except Exception:
        logger.info(f"'{args.collection}' does not exist — nothing to migrate. "
                    f"A fresh hybrid collection will be created on next app startup.")
        return

    vectors_config = info.config.params.vectors
    if isinstance(vectors_config, dict) and DENSE_VECTOR_NAME in vectors_config:
        logger.info(f"'{args.collection}' already uses the named hybrid schema — nothing to do.")
        return

    point_count = info.points_count
    logger.info(f"'{args.collection}' has {point_count} points, not yet on the hybrid schema.")

    # Distinguish "'{collection}' IS a physical collection" from "'{collection}'
    # is currently an alias pointing at one" — determines what the cutover
    # step needs to delete vs simply repoint. get_collections() only ever
    # lists physical collections (verified live), never aliases.
    physical_names = {c.name for c in client.get_collections().collections}
    if args.collection in physical_names:
        old_physical_name = args.collection
        is_alias = False
    else:
        matches = [a.collection_name for a in client.get_aliases().aliases if a.alias_name == args.collection]
        if not matches:
            raise RuntimeError(
                f"'{args.collection}' is neither a physical collection nor a known alias — "
                f"cannot determine what to migrate."
            )
        old_physical_name = matches[0]
        is_alias = True

    new_physical_name = f"{old_physical_name}_hybrid_{int(time.time())}"

    if not args.yes:
        verify_desc = ("a full payload check of every point" if args.full_verify
                        else f"a {args.verify_sample}-point payload sample")
        confirm = input(
            f"This will create '{new_physical_name}', copy all {point_count} points into it with "
            f"the hybrid schema, verify it (point count + schema + {verify_desc}), then atomically "
            f"point '{args.collection}' at it and delete the old physical collection "
            f"'{old_physical_name}'. An exclusive lock blocks all API uploads/deletes/folder changes "
            f"for the whole operation (see lock.exclusive_lock's docstring) — this should be fast, "
            f"since no re-embedding happens, but make sure you've run ./backup_qdrant.sh first "
            f"regardless. Continue? [y/N] "
        )
        if confirm.strip().lower() != "y":
            logger.info("Aborted.")
            return

    from lock import exclusive_lock

    logger.info(f"Acquiring exclusive maintenance lock (waits up to "
                f"{args.exclusive_lock_wait_seconds}s for in-flight API requests to finish)...")
    try:
        cm = exclusive_lock(wait_seconds=args.exclusive_lock_wait_seconds)
        cm.__enter__()
    except TimeoutError as e:
        logger.error(str(e))
        raise SystemExit(1)
    try:
        _migrate_locked(client, args, old_physical_name, new_physical_name, is_alias, point_count)
    finally:
        cm.__exit__(None, None, None)


def _migrate_locked(client, args, old_physical_name, new_physical_name, is_alias, point_count):
    from qdrant_client.models import (
        PointStruct, CreateAliasOperation, CreateAlias, DeleteAliasOperation, DeleteAlias,
    )
    from vector_db.qdrant_client import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, VectorStore
    from vector_db.sparse_encoder import build_sparse_vector

    # The count above was read BEFORE acquiring the lock — it has to be:
    # the confirmation prompt needs it, and that prompt must not itself run
    # while blocking the whole API on a human at a terminal. Re-read it now
    # that nothing can mutate the old collection from here on, so the
    # verification below compares against a number this run can actually
    # guarantee.
    current_info = client.get_collection(old_physical_name)
    if current_info.points_count != point_count:
        logger.warning(f"Point count changed from {point_count} to {current_info.points_count} "
                        f"between the pre-lock check and acquiring the exclusive lock — proceeding "
                        f"with the current, now-frozen count.")
        point_count = current_info.points_count

    vectors_config = current_info.config.params.vectors
    if isinstance(vectors_config, dict):
        # Not the documented "old single-unnamed-vector" case this script
        # targets — fall back to sampling one point's actual vector.
        sample, _ = client.scroll(collection_name=old_physical_name, limit=1, with_vectors=True)
        if not sample:
            raise RuntimeError(f"'{old_physical_name}' has 0 points and an unrecognized vectors "
                                f"config — cannot infer vector size.")
        v = sample[0].vector
        vector_size = len(v) if isinstance(v, list) else len(next(iter(v.values())))
    else:
        vector_size = vectors_config.size

    # Reuses the SAME already-connected `client` (not a fresh VectorStore(...)
    # -> its own new QdrantClient) — one Qdrant connection for the whole
    # script instead of two, and this object.__new__ + manual attribute
    # construction is exactly VectorStore.create_collection()'s own
    # _ensure_payload_indexes() call, which the migrated collection needs
    # the same as any other (document_id/folder/chunk_index/case_number/...
    # filters in hybrid_search rely on them).
    new_store = object.__new__(VectorStore)
    new_store.client = client
    new_store.collection = new_physical_name
    new_store.create_collection(vector_size=vector_size)
    logger.info(f"Created '{new_physical_name}' with the hybrid schema (vector_size={vector_size}).")

    logger.info("Streaming points old -> new (one scroll batch at a time, "
                "never the whole corpus held in memory)...")
    doc_meta = {}
    copied = 0
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=old_physical_name, limit=SCROLL_BATCH, offset=offset,
            with_payload=True, with_vectors=True,
        )
        if batch:
            new_points = []
            for point in batch:
                payload = point.payload or {}
                text = payload.get("text", "")
                dense_vector = point.vector if isinstance(point.vector, list) else point.vector.get("", [])
                new_points.append(PointStruct(
                    id=point.id,
                    vector={DENSE_VECTOR_NAME: dense_vector, SPARSE_VECTOR_NAME: build_sparse_vector(text)},
                    payload=payload,
                ))
                doc_id = payload.get("document_id", "")
                if doc_id:
                    meta = doc_meta.setdefault(doc_id, {
                        "doc_id": doc_id,
                        "filename": payload.get("filename", "unknown"),
                        "pages": payload.get("pages", 0),
                        "chunks": 0,
                        "size_kb": payload.get("size_kb", 0),
                        "metadata": {},
                        "folder": payload.get("folder", ""),
                        "format": payload.get("format", "pdf"),
                    })
                    meta["chunks"] += 1
            client.upsert(collection_name=new_physical_name, points=new_points)
            copied += len(new_points)
            logger.info(f"  copied {copied}/{point_count}")
        if next_offset is None:
            break
        offset = next_offset
    logger.info(f"Copied {copied} points into '{new_physical_name}'.")

    _verify_before_cutover(client, old_physical_name, new_physical_name, point_count, args)

    logger.info(f"Cutting over: '{args.collection}' -> '{new_physical_name}'...")
    if is_alias:
        # Repointing an existing alias is one atomic call — no gap where
        # the name resolves to nothing.
        client.update_collection_aliases(change_aliases_operations=[
            DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=args.collection)),
            CreateAliasOperation(create_alias=CreateAlias(
                collection_name=new_physical_name, alias_name=args.collection)),
        ])
    else:
        # '{args.collection}' is currently a real physical collection — free
        # the name, then claim it as an alias. Qdrant has no atomic rename,
        # so there's an unavoidable instant between these two calls where
        # the name resolves to nothing at all: any MUTATION is blocked out
        # for the whole operation by the exclusive lock, but a read-only
        # /query landing in exactly that instant would fail. This gap only
        # exists on the FIRST migration off a bare physical-collection name
        # — every migration after this one goes through the is_alias branch
        # above instead, which has none.
        client.delete_collection(old_physical_name)
        client.update_collection_aliases(change_aliases_operations=[
            CreateAliasOperation(create_alias=CreateAlias(
                collection_name=new_physical_name, alias_name=args.collection)),
        ])
    logger.info(f"Cutover complete: '{args.collection}' now points at '{new_physical_name}'.")

    logger.info(f"Backfilling {len(doc_meta)} documents into Postgres...")
    _backfill_postgres(args.postgres_url, doc_meta)
    logger.info("Migration complete.")


def _verify_before_cutover(client, old_physical_name, new_physical_name, point_count, args):
    """Count, schema, and payload verification of the NEW collection against
    the OLD one — all BEFORE the cutover touches either. Raises (aborting
    the whole run, leaving the OLD collection untouched and the NEW one in
    place for inspection) on any mismatch."""
    from vector_db.qdrant_client import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

    new_info = client.get_collection(new_physical_name)
    if new_info.points_count != point_count:
        raise RuntimeError(
            f"VERIFICATION FAILED: '{old_physical_name}' has {point_count} points but "
            f"'{new_physical_name}' has {new_info.points_count}. Refusing to cut over — "
            f"'{old_physical_name}' is untouched; '{new_physical_name}' is left in place for "
            f"inspection (delete it manually once you've diagnosed the discrepancy)."
        )

    new_vectors = new_info.config.params.vectors
    new_sparse = new_info.config.params.sparse_vectors
    if not (isinstance(new_vectors, dict) and DENSE_VECTOR_NAME in new_vectors
            and new_sparse and SPARSE_VECTOR_NAME in new_sparse):
        raise RuntimeError(
            f"VERIFICATION FAILED: '{new_physical_name}' does not have the expected named "
            f"dense+sparse hybrid schema (got vectors={new_vectors!r} sparse_vectors={new_sparse!r}). "
            f"Refusing to cut over."
        )
    logger.info(f"Count OK ({point_count}), schema OK (named dense+sparse hybrid).")

    if args.full_verify:
        logger.info("Full payload verification: comparing EVERY point, old vs new...")
        mismatches = _verify_payloads_full(client, old_physical_name, new_physical_name, point_count)
    else:
        logger.info(f"Sampling {args.verify_sample} points for payload verification...")
        sample, _ = client.scroll(
            collection_name=old_physical_name, limit=args.verify_sample,
            with_payload=True, with_vectors=False,
        )
        mismatches = _verify_payloads_batch(client, new_physical_name, sample)

    if mismatches:
        raise RuntimeError(
            f"VERIFICATION FAILED: {len(mismatches)} payload mismatch(es) between old and new — "
            f"e.g. {mismatches[:5]}. Refusing to cut over. '{old_physical_name}' is untouched; "
            f"'{new_physical_name}' is left in place for inspection."
        )
    logger.info("Payload verification OK.")


def _verify_payloads_batch(client, new_collection, old_points):
    """Compares a list of already-scrolled old points against their
    same-ID counterparts in new_collection (point IDs are preserved as-is
    by the copy, so this is a direct ID lookup, not a search). Returns a
    list of (point_id, reason) mismatches — empty means everything checked
    matched exactly."""
    if not old_points:
        return []
    ids = [p.id for p in old_points]
    new_points = client.retrieve(collection_name=new_collection, ids=ids, with_payload=True)
    new_by_id = {p.id: (p.payload or {}) for p in new_points}
    mismatches = []
    for old_point in old_points:
        if old_point.id not in new_by_id:
            mismatches.append((old_point.id, "missing in new collection"))
        elif new_by_id[old_point.id] != (old_point.payload or {}):
            mismatches.append((old_point.id, "payload differs"))
    return mismatches


def _verify_payloads_full(client, old_collection, new_collection, point_count):
    mismatches = []
    checked = 0
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=old_collection, limit=SCROLL_BATCH, offset=offset,
            with_payload=True, with_vectors=False,
        )
        mismatches.extend(_verify_payloads_batch(client, new_collection, batch))
        checked += len(batch)
        logger.info(f"  verified {checked}/{point_count}")
        if next_offset is None:
            break
        offset = next_offset
    return mismatches


def _backfill_postgres(postgres_url, doc_meta):
    import json
    import psycopg2

    conn = psycopg2.connect(postgres_url)
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


if __name__ == "__main__":
    main()
