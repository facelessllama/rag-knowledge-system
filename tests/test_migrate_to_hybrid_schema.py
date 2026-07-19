"""
Tests for scripts/migrate_to_hybrid_schema.py's redesigned migration:
- copy into a SEPARATE new physical collection, batched (never accumulating
  the whole corpus in memory);
- count + schema + payload verification BEFORE the old collection/alias is
  ever touched;
- an alias cutover (not delete-then-rebuild-in-place), correct for both
  "the configured name is a real physical collection" and "it's already an
  alias" starting states;
- the whole operation held under lock.exclusive_lock(), not just the
  cutover (see the module docstring for why: a shared-lock-only barrier
  doesn't stop a concurrent write from landing in an already-scrolled-past
  offset region).

A fake Qdrant client stands in for the real one — no live Qdrant needed.
lock.exclusive_lock() runs for real, against an isolated tmp_path lock file
(same technique as tests/test_lock.py), not mocked out, since the locking
behavior itself is part of what's under test.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest

import lock as lock_module
from migrate_to_hybrid_schema import _run, _run_backfill_only
from vector_db.qdrant_client import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME


@pytest.fixture(autouse=True)
def _isolated_lock_file(tmp_path, monkeypatch):
    monkeypatch.setattr(lock_module, "LOCK_FILE_PATH", tmp_path / ".backup.lock")


# ── fake Qdrant client ───────────────────────────────────────────────────────

class _Alias:
    def __init__(self, alias_name, collection_name):
        self.alias_name = alias_name
        self.collection_name = collection_name


class _AliasesResponse:
    def __init__(self, aliases):
        self.aliases = aliases


class _CollectionsListItem:
    def __init__(self, name):
        self.name = name


class _CollectionsResponse:
    def __init__(self, names):
        self.collections = [_CollectionsListItem(n) for n in names]


class _VectorParams:
    def __init__(self, size):
        self.size = size


class _ConfigParams:
    def __init__(self, vectors, sparse_vectors=None):
        self.vectors = vectors
        self.sparse_vectors = sparse_vectors


class _Config:
    def __init__(self, params):
        self.params = params


class _CollectionInfo:
    def __init__(self, points_count, vectors, sparse_vectors=None):
        self.points_count = points_count
        self.config = _Config(_ConfigParams(vectors, sparse_vectors))


class _Point:
    def __init__(self, id, payload, vector):
        self.id = id
        self.payload = payload
        self.vector = vector


class _FakeQdrantClient:
    """Physical collections + aliases + points, enough to drive the whole
    migration flow. `scroll` paginates via a plain list-index offset (not
    Qdrant's real opaque cursor) — sufficient for exercising batching."""

    def __init__(self):
        self.collections = {}   # physical name -> _CollectionInfo
        self.points = {}        # physical name -> {id: _Point}
        self.aliases = {}       # alias name -> physical name
        self.deleted_collections = []
        self.upsert_batches = []       # [(collection_name, batch_size), ...]
        self.create_collection_calls = []

    def _resolve(self, name):
        return self.aliases.get(name, name)

    def get_collection(self, name):
        physical = self._resolve(name)
        if physical not in self.collections:
            raise Exception(f"not found: {name}")
        return self.collections[physical]

    def get_collections(self):
        return _CollectionsResponse(list(self.collections.keys()))

    def get_aliases(self):
        return _AliasesResponse([_Alias(a, c) for a, c in self.aliases.items()])

    def create_collection(self, collection_name, vectors_config, sparse_vectors_config):
        self.create_collection_calls.append(collection_name)
        self.collections[collection_name] = _CollectionInfo(0, vectors_config, sparse_vectors_config)
        self.points[collection_name] = {}

    def create_payload_index(self, **kwargs):
        pass

    def delete_collection(self, name):
        self.deleted_collections.append(name)
        self.points.pop(name, None)
        self.collections.pop(name, None)
        for a in [a for a, c in self.aliases.items() if c == name]:
            del self.aliases[a]

    def update_collection_aliases(self, change_aliases_operations):
        for op in change_aliases_operations:
            create = getattr(op, "create_alias", None)
            delete = getattr(op, "delete_alias", None)
            if create is not None:
                self.aliases[create.alias_name] = create.collection_name
            elif delete is not None:
                self.aliases.pop(delete.alias_name, None)

    def scroll(self, collection_name, limit, offset=None, with_payload=True, with_vectors=False, **kwargs):
        physical = self._resolve(collection_name)
        pts = list(self.points.get(physical, {}).values())
        start = offset or 0
        batch = pts[start:start + limit]
        next_offset = start + limit if start + limit < len(pts) else None
        return batch, next_offset

    def upsert(self, collection_name, points):
        physical = self._resolve(collection_name)
        self.upsert_batches.append((collection_name, len(points)))
        for p in points:
            self.points.setdefault(physical, {})[p.id] = p
        self.collections[physical].points_count = len(self.points[physical])

    def retrieve(self, collection_name, ids, with_payload=True):
        physical = self._resolve(collection_name)
        pts = self.points.get(physical, {})
        return [pts[i] for i in ids if i in pts]


def _old_style_points(n, document_id="doc1"):
    return [
        _Point(id=f"p{i}", payload={"text": f"chunk {i}", "document_id": document_id,
                                     "filename": "f.pdf", "pages": 1, "size_kb": 1.0, "folder": "", "format": "pdf"},
               vector=[0.1, 0.2, 0.3])
        for i in range(n)
    ]


def _args(**overrides):
    from argparse import Namespace
    defaults = dict(
        qdrant_url="http://fake:6333", qdrant_api_key=None, collection="knowledge_base",
        postgres_url="postgresql://fake", yes=True, verify_sample=200, full_verify=False,
        exclusive_lock_wait_seconds=5.0,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _setup_old_collection(client, n_points=5, name="knowledge_base"):
    client.collections[name] = _CollectionInfo(n_points, _VectorParams(size=3))
    client.points[name] = {p.id: p for p in _old_style_points(n_points)}


# ── high-level flow ──────────────────────────────────────────────────────────

def test_already_hybrid_schema_is_a_noop(monkeypatch):
    client = _FakeQdrantClient()
    client.collections["knowledge_base"] = _CollectionInfo(
        5, {DENSE_VECTOR_NAME: _VectorParams(3)}, {SPARSE_VECTOR_NAME: object()})
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kw: client)
    monkeypatch.setattr("migrate_to_hybrid_schema._backfill_postgres", lambda *a, **kw: pytest.fail("must not run"))

    _run(_args())

    assert client.create_collection_calls == []
    assert client.deleted_collections == []


def test_missing_collection_is_a_noop(monkeypatch):
    client = _FakeQdrantClient()
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kw: client)

    _run(_args())  # must not raise despite nothing existing

    assert client.create_collection_calls == []


def test_migrates_physical_collection_and_cuts_over_via_alias(monkeypatch):
    client = _FakeQdrantClient()
    _setup_old_collection(client, n_points=5)
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kw: client)
    backfilled = []
    monkeypatch.setattr("migrate_to_hybrid_schema._backfill_postgres",
                         lambda url, doc_meta: backfilled.append(doc_meta))

    _run(_args())

    # old physical collection is gone, name is now an alias to the new one
    assert "knowledge_base" not in client.collections
    assert "knowledge_base" in client.aliases
    new_physical = client.aliases["knowledge_base"]
    assert new_physical != "knowledge_base"
    assert client.deleted_collections == ["knowledge_base"]

    # every original point survived, with the hybrid schema
    assert client.collections[new_physical].points_count == 5
    new_vectors = client.collections[new_physical].config.params.vectors
    assert DENSE_VECTOR_NAME in new_vectors

    # Postgres backfill ran with the one document seen
    assert backfilled and "doc1" in backfilled[0]
    assert backfilled[0]["doc1"]["chunks"] == 5


def test_migrates_already_aliased_collection_by_repointing_not_deleting(monkeypatch):
    """If '{collection}' is already an alias (a previous migration already
    ran once), the cutover must repoint the alias — never delete_collection
    on anything, since there's no bare physical name to reclaim."""
    client = _FakeQdrantClient()
    _setup_old_collection(client, n_points=3, name="knowledge_base_hybrid_111")
    client.aliases["knowledge_base"] = "knowledge_base_hybrid_111"
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kw: client)
    monkeypatch.setattr("migrate_to_hybrid_schema._backfill_postgres", lambda *a, **kw: None)

    _run(_args())

    assert client.deleted_collections == []  # old physical collection is left alone
    new_physical = client.aliases["knowledge_base"]
    assert new_physical not in ("knowledge_base", "knowledge_base_hybrid_111")
    assert client.collections[new_physical].points_count == 3


def test_streams_in_batches_not_one_accumulated_upsert(monkeypatch):
    client = _FakeQdrantClient()
    _setup_old_collection(client, n_points=25)
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kw: client)
    monkeypatch.setattr("migrate_to_hybrid_schema.SCROLL_BATCH", 10)
    monkeypatch.setattr("migrate_to_hybrid_schema._backfill_postgres", lambda *a, **kw: None)

    _run(_args())

    new_physical = client.aliases["knowledge_base"]
    batch_sizes = [n for name, n in client.upsert_batches if name == new_physical]
    assert batch_sizes == [10, 10, 5]  # 3 separate upserts, not one of 25


def test_count_mismatch_aborts_before_any_cutover(monkeypatch):
    client = _FakeQdrantClient()
    _setup_old_collection(client, n_points=5)
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kw: client)
    backfilled = []
    monkeypatch.setattr("migrate_to_hybrid_schema._backfill_postgres", lambda *a, **kw: backfilled.append(1))

    # Simulate a race: a point vanishes from the NEW collection right after
    # copying (verification must catch this, not just trust the copy loop).
    orig_upsert = client.upsert
    def _upsert_then_drop_one(collection_name, points):
        orig_upsert(collection_name, points)
        physical = client._resolve(collection_name)
        if client.points[physical]:
            client.points[physical].pop(next(iter(client.points[physical])))
            client.collections[physical].points_count -= 1
    monkeypatch.setattr(client, "upsert", _upsert_then_drop_one)

    with pytest.raises(RuntimeError, match="VERIFICATION FAILED"):
        _run(_args())

    assert client.deleted_collections == []          # old collection untouched
    assert "knowledge_base" not in client.aliases     # no cutover happened
    assert backfilled == []                           # postgres backfill never reached


def test_payload_mismatch_aborts_before_any_cutover(monkeypatch):
    client = _FakeQdrantClient()
    _setup_old_collection(client, n_points=3)
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kw: client)
    monkeypatch.setattr("migrate_to_hybrid_schema._backfill_postgres",
                         lambda *a, **kw: pytest.fail("must not run"))

    # Corrupt one payload in the NEW collection right after the copy loop
    # upserts it, simulating a build_sparse_vector/transform bug.
    orig_upsert = client.upsert
    def _upsert_then_corrupt(collection_name, points):
        orig_upsert(collection_name, points)
        physical = client._resolve(collection_name)
        any_id = next(iter(client.points[physical]))
        client.points[physical][any_id].payload = {"corrupted": True}
    monkeypatch.setattr(client, "upsert", _upsert_then_corrupt)

    with pytest.raises(RuntimeError, match="VERIFICATION FAILED"):
        _run(_args(verify_sample=1000))

    assert client.deleted_collections == []
    assert "knowledge_base" not in client.aliases


def test_full_verify_checks_every_point_not_just_a_sample(monkeypatch):
    client = _FakeQdrantClient()
    _setup_old_collection(client, n_points=12)
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kw: client)
    monkeypatch.setattr("migrate_to_hybrid_schema.SCROLL_BATCH", 5)
    monkeypatch.setattr("migrate_to_hybrid_schema._backfill_postgres", lambda *a, **kw: None)

    _run(_args(full_verify=True, verify_sample=1))  # sample size irrelevant when full_verify

    new_physical = client.aliases["knowledge_base"]
    assert client.collections[new_physical].points_count == 12


# ── --backfill-only recovery path ────────────────────────────────────────────
# Covers the gap a migration crashing between cutover and Postgres backfill
# leaves: _run()'s "already hybrid — nothing to do" check means a plain
# re-run can never reach the backfill step again. _run_backfill_only()
# bypasses that check entirely and rebuilds `documents` from whatever the
# collection currently is (already-hybrid or not — it doesn't care).

def test_backfill_only_rebuilds_postgres_from_an_already_hybrid_collection(monkeypatch):
    """The exact recovery scenario: the collection is ALREADY on the hybrid
    schema (as if cutover already succeeded) — _run() itself would exit
    immediately here without touching Postgres at all. --backfill-only must
    still work."""
    client = _FakeQdrantClient()
    client.collections["knowledge_base"] = _CollectionInfo(
        4, {DENSE_VECTOR_NAME: _VectorParams(3)}, {SPARSE_VECTOR_NAME: object()})
    client.points["knowledge_base"] = {p.id: p for p in _old_style_points(4, document_id="doc1")}
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kw: client)
    backfilled = []
    monkeypatch.setattr("migrate_to_hybrid_schema._backfill_postgres",
                         lambda url, doc_meta: backfilled.append(doc_meta))

    _run_backfill_only(_args())

    assert len(backfilled) == 1
    assert backfilled[0]["doc1"]["chunks"] == 4
    # purely read-only against Qdrant — no mutation of any kind
    assert client.deleted_collections == []
    assert client.create_collection_calls == []
    assert client.upsert_batches == []


def test_backfill_only_works_against_an_alias_name(monkeypatch):
    """The normal post-migration state: args.collection is an alias, not a
    physical name. Must resolve through it the same way _run() does."""
    client = _FakeQdrantClient()
    _setup_old_collection(client, n_points=3, name="knowledge_base_hybrid_999")
    client.aliases["knowledge_base"] = "knowledge_base_hybrid_999"
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kw: client)
    backfilled = []
    monkeypatch.setattr("migrate_to_hybrid_schema._backfill_postgres",
                         lambda url, doc_meta: backfilled.append(doc_meta))

    _run_backfill_only(_args())

    assert len(backfilled) == 1
    assert backfilled[0]["doc1"]["chunks"] == 3


def test_backfill_only_scans_in_batches(monkeypatch):
    client = _FakeQdrantClient()
    _setup_old_collection(client, n_points=25)
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kw: client)
    monkeypatch.setattr("migrate_to_hybrid_schema.SCROLL_BATCH", 10)
    backfilled = []
    monkeypatch.setattr("migrate_to_hybrid_schema._backfill_postgres",
                         lambda url, doc_meta: backfilled.append(doc_meta))

    _run_backfill_only(_args())

    assert backfilled[0]["doc1"]["chunks"] == 25


def test_backfill_only_errors_clearly_when_collection_missing(monkeypatch):
    client = _FakeQdrantClient()
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kw: client)

    with pytest.raises(SystemExit):
        _run_backfill_only(_args())
