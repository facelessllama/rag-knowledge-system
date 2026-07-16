#!/usr/bin/env python3
"""
One-time migration: widen doc_id from a truncated 8-hex-char uuid4()
prefix (32 bits of entropy — str(uuid.uuid4())[:8]) to a full uuid4()
(122 bits), stored as Postgres native UUID (psycopg2 returns UUID columns
as plain str by default in this environment — verified, so no other code
needs to change). At 57,000 documents — a scale this project has actually
been tested at, see eval/README.md's scale ladder — the birthday-paradox
collision probability at 32 bits of entropy is ~31.5%. A collision would
silently overwrite one document's metadata (documents' ON CONFLICT DO
UPDATE) and mix its Qdrant points with another's, since both share a
document_id and a chunk_id prefix.

doc_id touches three systems, updated in this order per document:
  1. Qdrant: recompute/fix `document_id` and `chunk_id` (chunk_id embeds
     doc_id as a literal prefix — see ingestion/chunker.py:
     f"{doc_id}_p{page_num}_c{chunk_index}") on every point found under
     EITHER old_id or new_id — not just old_id — and independently checks
     each field rather than assuming "found under old_id" implies both
     fields need updating. This is what makes a crash mid-document
     recoverable: a naive version that only ever searched old_id would, on
     re-run after a crash between the document_id update and the
     chunk_id updates, find zero points under old_id (document_id already
     flipped), conclude the document was already done, and let step 2
     commit the new id in Postgres — permanently stranding whatever
     chunk_ids hadn't been fixed yet, since nothing would ever search for
     them under either id again. Fully paginated (a single unpaginated
     scroll() silently processed only the first ~1000 points of a larger
     document); raises if any point still references old_id once finished,
     so a document is never silently left half-migrated.
  2. Postgres: UPDATE documents.doc_id (cascades to document_pages.
     document_id automatically via the FK's ON UPDATE CASCADE — added by
     --prepare-schema below) and file_hashes.doc_id (no FK — updated
     explicitly) in one transaction.
  3. Filesystem: rename uploads/{old}_{filename} -> uploads/{new}_{filename}.

Idempotent / resumable: the old->new UUID mapping is generated once and
persisted to migration_map.json next to this script — a re-run after any
interruption reads the SAME mapping back (never re-rolls new UUIDs for a
document already assigned one) and, for each document, independently
checks which of the three systems still needs updating rather than
assuming all-or-nothing, so a document that got as far as step 2 before a
crash simply finishes step 3 on the next run instead of getting a second,
different UUID.

Schema prep (run once, safe to re-run): widens doc_id/document_id/
file_hashes.doc_id from VARCHAR(8) to VARCHAR(36) — old short values stay
valid, VARCHAR only caps a maximum length, doesn't pad — and adds ON
UPDATE CASCADE to document_pages' FK. Both are required before any row's
doc_id can actually be updated. Run --tighten-schema only after every
document has been migrated (all values are valid UUID-format strings by
then), to convert the columns to native UUID.

Usage:
    source venv/bin/activate
    python scripts/migrate_doc_ids_to_uuid.py --prepare-schema   # once, first
    python scripts/migrate_doc_ids_to_uuid.py --dry-run          # report only
    python scripts/migrate_doc_ids_to_uuid.py                    # migrate all documents
    python scripts/migrate_doc_ids_to_uuid.py --tighten-schema   # once, after migration succeeds
"""
import argparse
import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("migrate_doc_ids_to_uuid")

MAP_PATH = Path(__file__).resolve().parent / "migration_map.json"

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def is_already_migrated(doc_id: str) -> bool:
    return bool(_UUID_RE.match(doc_id))


def load_or_build_mapping(old_ids: list[str]) -> dict[str, str]:
    mapping = {}
    if MAP_PATH.exists():
        mapping = json.loads(MAP_PATH.read_text())
    changed = False
    for old_id in old_ids:
        if old_id not in mapping:
            mapping[old_id] = str(uuid.uuid4())
            changed = True
    if changed:
        MAP_PATH.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    return mapping


def prepare_schema(conn):
    cur = conn.cursor()
    cur.execute("ALTER TABLE file_hashes ALTER COLUMN doc_id TYPE VARCHAR(36)")
    cur.execute("ALTER TABLE documents ALTER COLUMN doc_id TYPE VARCHAR(36)")
    cur.execute("ALTER TABLE document_pages ALTER COLUMN document_id TYPE VARCHAR(36)")
    cur.execute("ALTER TABLE document_pages DROP CONSTRAINT document_pages_document_id_fkey")
    cur.execute("""
        ALTER TABLE document_pages
        ADD CONSTRAINT document_pages_document_id_fkey
        FOREIGN KEY (document_id) REFERENCES documents(doc_id)
        ON DELETE CASCADE ON UPDATE CASCADE
    """)
    conn.commit()
    logger.info("Schema prepared: columns widened to VARCHAR(36), FK now ON UPDATE CASCADE.")


def tighten_schema(conn):
    cur = conn.cursor()
    cur.execute("SELECT doc_id FROM documents WHERE doc_id !~ '^[0-9a-fA-F-]{36}$'")
    unmigrated = cur.fetchall()
    if unmigrated:
        raise RuntimeError(
            f"{len(unmigrated)} documents still have a non-UUID doc_id — run the migration "
            f"(without flags) until it reports 0 remaining before tightening the schema."
        )
    # The FK must be dropped first — Postgres checks type compatibility as
    # soon as either side changes, and won't allow documents.doc_id (about
    # to become uuid) and document_pages.document_id (still varchar) to
    # differ even mid-transaction.
    cur.execute("ALTER TABLE document_pages DROP CONSTRAINT document_pages_document_id_fkey")
    cur.execute("ALTER TABLE documents ALTER COLUMN doc_id TYPE UUID USING doc_id::uuid")
    cur.execute("ALTER TABLE document_pages ALTER COLUMN document_id TYPE UUID USING document_id::uuid")
    cur.execute("ALTER TABLE file_hashes ALTER COLUMN doc_id TYPE UUID USING doc_id::uuid")
    cur.execute("""
        ALTER TABLE document_pages
        ADD CONSTRAINT document_pages_document_id_fkey
        FOREIGN KEY (document_id) REFERENCES documents(doc_id)
        ON DELETE CASCADE ON UPDATE CASCADE
    """)
    conn.commit()
    logger.info("Schema tightened: doc_id/document_id columns are now native UUID.")


def _scroll_by_document_id(client, collection, doc_id):
    """Paginated — a document with more than one scroll page of chunks
    (previously fetched with a single limit=1000, no next_offset follow-up)
    would otherwise migrate only its first page, leaving the rest on the
    old document_id/chunk_id while Postgres and the filename already moved
    to the new one."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    points = []
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]),
            limit=500, offset=offset, with_payload=True, with_vectors=False,
        )
        points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset
    return points


def migrate_qdrant(client, collection, old_id, new_id, dry_run):
    """Searches under BOTH old_id and new_id, and independently verifies/
    fixes document_id and chunk_id per point, rather than assuming a point
    found under old_id needs both fields updated and a point not found
    under old_id is already fully done. A crash between the batched
    document_id update and the per-point chunk_id updates used to be
    unrecoverable: a re-run searching only old_id would find zero points
    (document_id already flipped), conclude "already migrated", and let
    migrate_postgres() commit the new id — permanently stranding whatever
    chunk_ids hadn't been fixed yet, since nothing would ever search for
    them under either id again."""
    points = _scroll_by_document_id(client, collection, old_id)
    points += _scroll_by_document_id(client, collection, new_id)

    if not points:
        return 0  # nothing to migrate (e.g. a zero-chunk document)

    if dry_run:
        return len(points)

    stale_document_id = [p.id for p in points if (p.payload or {}).get("document_id") != new_id]
    if stale_document_id:
        client.set_payload(collection_name=collection, payload={"document_id": new_id}, points=stale_document_id)

    for p in points:
        payload = p.payload or {}
        expected_chunk_id = f"{new_id}_p{payload.get('page_num')}_c{payload.get('chunk_index')}"
        if payload.get("chunk_id") != expected_chunk_id:
            client.set_payload(collection_name=collection, payload={"chunk_id": expected_chunk_id}, points=[p.id])

    # Belt-and-braces: re-fetch fresh (not the `points` list already held in
    # memory, which could itself reflect a set_payload that silently failed
    # to apply) and verify BOTH fields on every point now living under
    # new_id, not just "nothing is left under old_id" — a point whose
    # document_id update succeeded but whose chunk_id update silently
    # didn't would already be invisible to an old_id-only check (it's
    # correctly not found there) while still carrying a stale chunk_id.
    final_points = _scroll_by_document_id(client, collection, new_id)
    bad_chunk_ids = [
        p.id for p in final_points
        if (p.payload or {}).get("chunk_id")
        != f"{new_id}_p{(p.payload or {}).get('page_num')}_c{(p.payload or {}).get('chunk_index')}"
    ]
    if bad_chunk_ids:
        raise RuntimeError(f"{len(bad_chunk_ids)} point(s) under {new_id} still have a stale chunk_id after migration")

    remaining_old = _scroll_by_document_id(client, collection, old_id)
    if remaining_old:
        raise RuntimeError(f"{len(remaining_old)} point(s) still reference old document_id {old_id} after migration")

    return len(points)


def migrate_postgres(conn, old_id, new_id, dry_run):
    """Checks new_id FIRST, and always compares via doc_id::text — once
    --tighten-schema has converted the column to native UUID (the normal
    end state of this whole migration), comparing a non-UUID-shaped 8-char
    old_id directly against a UUID column (`doc_id = %s`) doesn't return
    "no rows", it raises "invalid input syntax for type uuid" outright.
    Casting the column to text instead is valid regardless of whether it's
    still VARCHAR or already UUID, and checking new_id up front means an
    already-fully-migrated document never even attempts the old_id
    comparison. Callers MUST conn.rollback() on any exception from this
    function before continuing the loop — psycopg2 leaves the whole
    transaction aborted after any failed statement, so every subsequent
    document on the same connection would otherwise fail too."""
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM documents WHERE doc_id::text = %s", (new_id,))
    if cur.fetchone():
        return False  # already migrated
    cur.execute("SELECT 1 FROM documents WHERE doc_id::text = %s", (old_id,))
    if not cur.fetchone():
        return False  # neither id present for this pair — nothing to do
    if dry_run:
        return True
    cur.execute("UPDATE documents SET doc_id = %s WHERE doc_id::text = %s", (new_id, old_id))
    cur.execute("UPDATE file_hashes SET doc_id = %s WHERE doc_id::text = %s", (new_id, old_id))
    conn.commit()
    return True


def migrate_file(upload_dir: Path, old_id, new_id, dry_run):
    old_file = next(upload_dir.glob(f"{old_id}_*"), None)
    if not old_file:
        return False  # already renamed (or never existed)
    new_name = new_id + old_file.name[len(old_id):]
    if dry_run:
        return True
    old_file.rename(upload_dir / new_name)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-schema", action="store_true", help="widen columns + FK cascade, run once first")
    ap.add_argument("--tighten-schema", action="store_true", help="VARCHAR(36) -> UUID, run once after migrating")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    from lock import run_locked
    run_locked(lambda: _run(args), logger)


def _run(args):
    import psycopg2

    from vector_db.qdrant_client import VectorStore

    postgres_url = os.getenv("POSTGRES_URL", "postgresql://raguser:ragpass@localhost:5432/ragdb")
    upload_dir = Path("uploads")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = os.getenv("QDRANT_COLLECTION", "knowledge_base")

    conn = psycopg2.connect(postgres_url)

    if args.prepare_schema:
        prepare_schema(conn)
        return
    if args.tighten_schema:
        tighten_schema(conn)
        return

    cur = conn.cursor()
    cur.execute("SELECT doc_id FROM documents ORDER BY doc_id")
    all_ids = [row[0] for row in cur.fetchall()]
    old_ids = [d for d in all_ids if not is_already_migrated(d)]

    mapping = load_or_build_mapping(old_ids)

    # Iterate the FULL persisted mapping — every old_id/new_id pair ever
    # assigned — not just doc_ids currently found un-migrated in Postgres.
    # A document whose Postgres row already committed the new UUID
    # (migrate_postgres succeeded) but whose file rename then failed would
    # otherwise vanish from `old_ids` on the very next run — Postgres no
    # longer has anything under the old id to find — even though the file
    # on disk is still sitting under its old name and would never get
    # picked up again. The mapping file remembers every pair permanently,
    # so a document that's done in one system but not another stays in
    # scope until all three independently agree.
    to_process = sorted(mapping.items())

    logger.info(f"{len(all_ids)} documents in Postgres — {len(old_ids)} not yet UUID-format there; "
                f"{len(to_process)} old_id/new_id pair(s) tracked overall (includes any left over "
                f"from a previous interrupted run){' (dry run)' if args.dry_run else ''}")
    if not to_process:
        logger.info("Nothing to do.")
        return

    vector_store = VectorStore(url=qdrant_url, collection=collection, api_key=os.getenv("QDRANT_API_KEY"))

    migrated, failed = 0, []
    for old_id, new_id in to_process:
        try:
            n_points = migrate_qdrant(vector_store.client, collection, old_id, new_id, args.dry_run)
            migrate_postgres(conn, old_id, new_id, args.dry_run)
            renamed = migrate_file(upload_dir, old_id, new_id, args.dry_run)
            logger.info(f"{'[DRY RUN] would migrate' if args.dry_run else 'migrated'} "
                        f"{old_id} -> {new_id} ({n_points} Qdrant points, file_renamed={renamed})")
            migrated += 1
        except Exception as e:
            logger.error(f"FAILED {old_id} -> {new_id}: {e}")
            failed.append((old_id, new_id, str(e)))
            # A failed statement leaves the WHOLE transaction aborted on this
            # connection — every subsequent document's queries would raise
            # "current transaction is aborted" otherwise, turning one bad
            # document into a total loss for the rest of the run.
            conn.rollback()

    logger.info("=" * 70)
    logger.info(f"{'[DRY RUN] ' if args.dry_run else ''}Done. migrated={migrated} failed={len(failed)}")
    for old_id, new_id, reason in failed:
        logger.error(f"  FAILED {old_id} -> {new_id}: {reason}")
    if failed:
        logger.error("Re-run this script (same command) to retry only the documents that failed — "
                     "the mapping file makes this safe to repeat.")


if __name__ == "__main__":
    main()
