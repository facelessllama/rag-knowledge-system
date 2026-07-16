"""
Tests for scripts/migrate_doc_ids_to_uuid.py — migrate_qdrant() (the part
that actually rewrites document_id/chunk_id in Qdrant, with an in-memory
mutable fake so these exercise the exact interleavings that make or break
resumability) and an orchestration-level test for the Postgres/filesystem
crash gap. No real infra needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest

from migrate_doc_ids_to_uuid import migrate_qdrant, migrate_postgres, migrate_file


class _FakePoint:
    def __init__(self, point_id, payload):
        self.id = point_id
        self.payload = dict(payload)


class _FakeQdrantClient:
    """Live, mutable point store — scroll() filters against CURRENT payload
    state (so a set_payload() call made earlier in the same migrate_qdrant()
    call is visible to a later scroll() searching a different id, exactly
    like the real thing), and paginates in `page_size`-sized batches
    regardless of the caller's requested limit, so a document with more
    points than one page exercises the next_offset loop rather than
    silently only covering the first page."""

    def __init__(self, points, page_size=2):
        self._points = {p.id: p for p in points}
        self.page_size = page_size
        self.set_payload_calls = []

    def scroll(self, collection_name, scroll_filter, limit, offset=None, with_payload=True, with_vectors=False):
        target = scroll_filter.must[0].match.value
        matches = sorted(
            (p for p in self._points.values() if p.payload.get("document_id") == target),
            key=lambda p: p.id,
        )
        ids = [p.id for p in matches]
        start = ids.index(offset) if offset is not None else 0
        page = matches[start:start + self.page_size]
        next_offset = matches[start + self.page_size].id if start + self.page_size < len(matches) else None
        return page, next_offset

    def set_payload(self, collection_name, payload, points):
        self.set_payload_calls.append((dict(payload), list(points)))
        for pid in points:
            self._points[pid].payload.update(payload)


def _point(pid, document_id, chunk_id, page_num=1, chunk_index=0):
    return _FakePoint(pid, {
        "document_id": document_id, "chunk_id": chunk_id,
        "page_num": page_num, "chunk_index": chunk_index,
    })


def test_migrate_qdrant_paginates_beyond_one_page():
    # 5 points under old_id, fake client pages 2 at a time — a single
    # unpaginated scroll(limit=1000) would still get all 5 in one real
    # Qdrant call for a fixture this small, so page_size=2 here is what
    # actually forces the next_offset loop to run more than once.
    points = [_point(f"p{i}", "old-id", f"old-id_p1_c{i}", chunk_index=i) for i in range(5)]
    client = _FakeQdrantClient(points, page_size=2)

    n = migrate_qdrant(client, "col", "old-id", "new-id", dry_run=False)

    assert n == 5
    for p in points:
        assert p.payload["document_id"] == "new-id"
        assert p.payload["chunk_id"] == f"new-id_p1_c{p.payload['chunk_index']}"


def test_migrate_qdrant_resumes_when_document_id_flipped_but_chunk_id_stale():
    """Simulates a crash between the batched document_id update and the
    per-point chunk_id updates: document_id is already new_id, but
    chunk_id still carries the old prefix. A version that only ever
    searched old_id would find nothing here, declare the document done,
    and let Postgres commit — permanently stranding this chunk_id."""
    points = [
        _point("p0", "new-id", "old-id_p1_c0", chunk_index=0),
        _point("p1", "new-id", "new-id_p1_c1", chunk_index=1),  # already fully fixed
    ]
    client = _FakeQdrantClient(points, page_size=10)

    n = migrate_qdrant(client, "col", "old-id", "new-id", dry_run=False)

    assert n == 2
    assert points[0].payload["chunk_id"] == "new-id_p1_c0"
    assert points[1].payload["chunk_id"] == "new-id_p1_c1"
    # The already-correct point must not have needed a document_id fix
    stale_calls = [c for c in client.set_payload_calls if "document_id" in c[0]]
    assert all("p1" not in ids for _, ids in stale_calls)


def test_migrate_qdrant_is_idempotent_when_already_fully_migrated():
    points = [_point("p0", "new-id", "new-id_p1_c0", chunk_index=0)]
    client = _FakeQdrantClient(points, page_size=10)

    n = migrate_qdrant(client, "col", "old-id", "new-id", dry_run=False)

    assert n == 1
    assert client.set_payload_calls == []  # nothing needed fixing — no writes at all


def test_migrate_qdrant_dry_run_makes_no_writes():
    points = [_point("p0", "old-id", "old-id_p1_c0", chunk_index=0)]
    client = _FakeQdrantClient(points, page_size=10)

    n = migrate_qdrant(client, "col", "old-id", "new-id", dry_run=True)

    assert n == 1
    assert client.set_payload_calls == []
    assert points[0].payload["document_id"] == "old-id"  # untouched


def test_migrate_qdrant_raises_if_a_point_is_left_on_old_id():
    """A point that set_payload() silently fails to update (Qdrant-side
    partial failure that doesn't raise) must not be shrugged off — the
    final re-check under old_id has to catch it and fail loudly rather
    than let the caller proceed to commit Postgres against an
    incompletely-migrated document."""
    class _StubbornClient(_FakeQdrantClient):
        def set_payload(self, collection_name, payload, points):
            # "p_stuck" never actually updates, as if the write silently
            # no-op'd or targeted the wrong point.
            points = [p for p in points if p != "p_stuck"]
            super().set_payload(collection_name, payload, points)

    points = [
        _point("p_stuck", "old-id", "old-id_p1_c0", chunk_index=0),
        _point("p_ok", "old-id", "old-id_p1_c1", chunk_index=1),
    ]
    client = _StubbornClient(points, page_size=10)

    with pytest.raises(RuntimeError, match="still reference old document_id"):
        migrate_qdrant(client, "col", "old-id", "new-id", dry_run=False)


def test_migrate_qdrant_raises_if_chunk_id_silently_fails_to_update():
    """Distinct from the document_id-stuck case above: here document_id
    updates fine (so the old_id-absence check alone would pass silently),
    but chunk_id on one point never actually applies. The final audit must
    fetch fresh state under new_id and check chunk_id there too, not just
    confirm nothing is left under old_id."""
    class _ChunkIdStubbornClient(_FakeQdrantClient):
        def set_payload(self, collection_name, payload, points):
            if "chunk_id" in payload and "p_stuck" in points:
                points = [p for p in points if p != "p_stuck"]
            super().set_payload(collection_name, payload, points)

    points = [
        _point("p_stuck", "old-id", "old-id_p1_c0", chunk_index=0),
        _point("p_ok", "old-id", "old-id_p1_c1", chunk_index=1),
    ]
    client = _ChunkIdStubbornClient(points, page_size=10)

    with pytest.raises(RuntimeError, match="stale chunk_id"):
        migrate_qdrant(client, "col", "old-id", "new-id", dry_run=False)

    # document_id itself DID update correctly — proving the failure is
    # specifically caught by the chunk_id audit, not a document_id check.
    assert points[0].payload["document_id"] == "new-id"
    assert points[0].payload["chunk_id"] == "old-id_p1_c0"  # never fixed


# ── Orchestration: crash after Postgres commit, before file rename ─────────

class _FakePgCursor:
    def __init__(self, store):
        self.store = store
        self._found = False

    def execute(self, query, params):
        if "SELECT 1 FROM documents" in query:
            self._found = params[0] in self.store["documents"]
        elif "UPDATE documents" in query:
            new_id, old_id = params
            if old_id in self.store["documents"]:
                self.store["documents"].remove(old_id)
                self.store["documents"].add(new_id)

    def fetchone(self):
        return (1,) if self._found else None


class _FakePgConn:
    def __init__(self, store):
        self.store = store

    def cursor(self):
        return _FakePgCursor(self.store)

    def commit(self):
        pass


def test_orchestration_resumes_file_rename_after_postgres_already_committed(tmp_path):
    """The exact crash the reviewer described: migrate_postgres() commits
    the new UUID, then the process dies before migrate_file() renames the
    file. main()'s old logic derived "what to migrate" from Postgres's
    CURRENT doc_id values — since Postgres has no row under old_id left to
    find, the document would silently drop out of scope on the next run
    and the file would stay under its old name forever. The fix (iterating
    the persisted mapping.json in full, not Postgres-derived old_ids) means
    this old_id/new_id pair gets revisited regardless of what Postgres
    already reflects — this test proves that revisit correctly finishes
    the file rename while leaving the (already-done) Postgres step a
    no-op, which is the invariant the orchestration fix relies on."""
    old_id, new_id = "old12345", "11111111-1111-1111-1111-111111111111"
    store = {"documents": {old_id}}
    conn = _FakePgConn(store)
    (tmp_path / f"{old_id}_report.pdf").write_text("hello")

    # --- "First run": Postgres step succeeds; crash happens right after,
    # before migrate_file() is ever called. ---
    did_update = migrate_postgres(conn, old_id, new_id, dry_run=False)
    assert did_update is True
    assert old_id not in store["documents"] and new_id in store["documents"]
    assert (tmp_path / f"{old_id}_report.pdf").exists()  # untouched — crash point

    # --- "Second run": orchestration revisits the SAME pair (this is what
    # iterating the persisted mapping achieves) ---
    did_update_again = migrate_postgres(conn, old_id, new_id, dry_run=False)
    assert did_update_again is False  # correctly a no-op — already migrated
    renamed = migrate_file(tmp_path, old_id, new_id, dry_run=False)
    assert renamed is True
    assert not (tmp_path / f"{old_id}_report.pdf").exists()
    assert (tmp_path / f"{new_id}_report.pdf").exists()


# ── migrate_postgres against an already-tightened (native UUID) column ──────

import re


def _is_valid_uuid(s):
    return bool(re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", s, re.IGNORECASE))


class _TightenedSchemaFakeCursor:
    """Simulates a documents table whose doc_id column is already native
    UUID — the end state --tighten-schema produces. A bare `doc_id = %s`
    comparison against a non-UUID-shaped parameter raises here exactly
    like real Postgres's "invalid input syntax for type uuid" (reproduced
    live against the actual database in review), while `doc_id::text = %s`
    never does regardless of the parameter's shape — this is what actually
    catches a regression back to the old query, which a fake that just
    records calls without caring about SQL content could not."""

    def __init__(self, conn):
        self.conn = conn

    def execute(self, query, params):
        if self.conn.aborted:
            raise Exception("current transaction is aborted, commands ignored until end of transaction block")
        try:
            compares_as_text = "doc_id::text" in query
            if "SELECT 1 FROM documents" in query:
                target = params[0]
                if not compares_as_text and not _is_valid_uuid(target):
                    raise Exception(f'invalid input syntax for type uuid: "{target}"')
                self._found = target in self.conn.ids
            elif "UPDATE documents" in query:
                new_id, old_id = params
                if not compares_as_text and not _is_valid_uuid(old_id):
                    raise Exception(f'invalid input syntax for type uuid: "{old_id}"')
                if old_id in self.conn.ids:
                    self.conn.ids.remove(old_id)
                self.conn.ids.add(new_id)
            elif "UPDATE file_hashes" in query:
                pass
        except Exception:
            self.conn.aborted = True
            raise

    def fetchone(self):
        return (1,) if getattr(self, "_found", False) else None


class _TightenedSchemaFakeConn:
    def __init__(self, existing_ids):
        self.ids = set(existing_ids)
        self.aborted = False

    def cursor(self):
        return _TightenedSchemaFakeCursor(self)

    def commit(self):
        pass

    def rollback(self):
        self.aborted = False


def test_migrate_postgres_works_against_already_tightened_uuid_column():
    """The exact regression this fixes: after --tighten-schema, doc_id is
    native UUID, and comparing an old 8-char id directly (`doc_id = %s`)
    against that column raises outright rather than returning "no rows" —
    migrate_postgres must use doc_id::text so this never happens."""
    old_id, new_id = "d2b91414", "11111111-1111-1111-1111-111111111111"
    conn = _TightenedSchemaFakeConn(existing_ids={old_id})

    result = migrate_postgres(conn, old_id, new_id, dry_run=False)

    assert result is True
    assert new_id in conn.ids
    assert old_id not in conn.ids


def test_migrate_postgres_checks_new_id_first_and_short_circuits():
    """Already-migrated (new_id present, old_id long gone) must return
    False cleanly."""
    old_id, new_id = "d2b91414", "11111111-1111-1111-1111-111111111111"
    conn = _TightenedSchemaFakeConn(existing_ids={new_id})  # old_id already gone

    result = migrate_postgres(conn, old_id, new_id, dry_run=False)

    assert result is False


def test_migrate_postgres_transaction_recovers_after_caller_rollback():
    """Reproduces the cascade the reviewer described: one query that
    raises poisons the WHOLE connection for every subsequent statement —
    real Postgres refuses every further command with "current transaction
    is aborted" until a rollback happens. Without the caller rolling back
    after catching an exception (which is what main()'s loop now does),
    the NEXT document's migrate_postgres call — and every one after it —
    would also fail, turning one bad document into a total loss for the
    rest of the run."""
    conn = _TightenedSchemaFakeConn(existing_ids={"old-id"})

    # Poison the transaction with an unrelated bad query (something else
    # going wrong elsewhere) — not migrate_postgres's own fault, since it
    # always uses ::text and would never trigger this itself.
    with pytest.raises(Exception):
        conn.cursor().execute("SELECT 1 FROM documents WHERE doc_id = %s", ("not-a-uuid",))

    # Without a rollback, even an otherwise-valid migrate_postgres call now fails.
    with pytest.raises(Exception, match="aborted"):
        migrate_postgres(conn, "old-id", "11111111-1111-1111-1111-111111111111", dry_run=False)

    conn.rollback()  # exactly what main()'s except block now does

    # Recovered — the same call now succeeds normally.
    result = migrate_postgres(conn, "old-id", "11111111-1111-1111-1111-111111111111", dry_run=False)
    assert result is True
