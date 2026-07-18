"""
Tests for the DB-state/tombstone fix to folder rename/move and document
delete not being atomic across Postgres + Qdrant + disk:

  - db_update_document_folder()/db_save_folder()/db_delete_folder()/
    db_rename_folder()/init_db() used to swallow every Postgres failure into
    a logged warning — callers had no way to tell a durable write from a
    silently-failed one, and went on to mutate documents_registry/
    folders_registry as if it had succeeded. They now raise; every
    caller below writes Postgres FIRST and only touches memory once that
    call returns without raising.
  - documents.status ('active'|'deleting') is a durable tombstone written
    BEFORE delete_document() touches Qdrant or the file on disk, so a crash
    mid-delete leaves a row a retry (calling DELETE again) or the startup
    sweep can find and finish, instead of a permanently orphaned file or a
    document that silently reappears after a partial failure.
  - documents.folder_synced tracks whether Qdrant's payload.folder is known
    to match Postgres's `folder` column — set FALSE atomically with the
    folder change itself, set back TRUE only once Qdrant confirms the
    write. rename_folder()'s per-document loop no longer aborts on the
    first Qdrant failure (the original bug: one exception left every
    document after it in iteration order untouched, even in memory).

Postgres/Qdrant are faked via monkeypatch — no live services required, same
approach as test_document_page_endpoint.py.
"""
from contextlib import contextmanager
import asyncio
import threading
import time

import pytest
from fastapi import HTTPException

import api.main as m


@pytest.fixture(autouse=True)
def _fresh_metadata_mutation_lock(monkeypatch):
    """_ReentrantAsyncLock binds to whatever event loop first acquires it and
    stays bound for its whole life — correct for production (one loop, one
    lock, for the process's entire run) but wrong across tests, since
    pytest-asyncio hands each async test function a fresh event loop while
    api.main's module-level `_metadata_mutation_lock` singleton survives
    unchanged between them. Swapping in a brand-new instance before every
    test (sync tests never touch it, so this is harmless overhead for them)
    means each test's first acquire binds fresh, instead of an earlier
    version of this class trying to detect and repair a stale binding
    itself — see _ReentrantAsyncLock's docstring in api/main.py."""
    monkeypatch.setattr(m, "_metadata_mutation_lock", m._ReentrantAsyncLock())


# ── fakes ─────────────────────────────────────────────────────────────────

def _failing_db_conn(message):
    class _FailingCursor:
        def execute(self, *args, **kwargs):
            raise RuntimeError(message)

    class _FailingConn:
        def cursor(self):
            return _FailingCursor()

    @contextmanager
    def _conn():
        yield _FailingConn()
    return _conn


class _FakeQdrantClient:
    """fail_doc_ids selectively fails set_payload/delete for specific
    doc_ids (extracted from the Filter's MatchValue); fail_always fails
    every call regardless of doc_id."""

    def __init__(self, fail_doc_ids=None, fail_always=False):
        self.fail_doc_ids = fail_doc_ids or set()
        self.fail_always = fail_always
        self.set_payload_calls = []
        self.delete_calls = []

    def _maybe_fail(self, doc_id):
        if self.fail_always or doc_id in self.fail_doc_ids:
            raise RuntimeError(f"simulated qdrant failure for {doc_id}")

    def set_payload(self, collection_name, payload, points):
        doc_id = points.must[0].match.value
        self.set_payload_calls.append((doc_id, dict(payload)))
        self._maybe_fail(doc_id)

    def delete(self, collection_name, points_selector):
        doc_id = points_selector.must[0].match.value
        self.delete_calls.append(doc_id)
        self._maybe_fail(doc_id)


class _FakeVectorStore:
    def __init__(self, client):
        self.client = client
        self.collection = "test_collection"


class _HangingQdrantClient:
    """Simulates the exact failure QDRANT_RECONCILE_TIMEOUT_SECONDS' wait_for
    backstop exists for: a call that never returns on its own (the
    client-level timeout somehow failing to fire). Runs on a real thread
    (via asyncio.to_thread in the code under test), so this must block with
    a real threading primitive, not just raise — release_event.set() at the
    end of a test lets the orphaned thread finish instead of leaking a
    thread across the test session."""

    def __init__(self):
        self.release_event = threading.Event()
        self.calls = 0

    def _block(self):
        self.calls += 1
        self.release_event.wait(timeout=5)

    def set_payload(self, collection_name, payload, points):
        self._block()

    def delete(self, collection_name, points_selector):
        self._block()


# ── db helpers no longer swallow (were: except Exception: logger.warning) ──

def test_db_update_document_folder_raises_on_failure(monkeypatch):
    monkeypatch.setattr(m, "db_conn", _failing_db_conn("simulated"))
    with pytest.raises(RuntimeError, match="simulated"):
        m.db_update_document_folder("d1", "NewFolder")


def test_db_save_folder_raises_on_failure(monkeypatch):
    monkeypatch.setattr(m, "db_conn", _failing_db_conn("simulated"))
    with pytest.raises(RuntimeError, match="simulated"):
        m.db_save_folder("NewFolder")


def test_db_delete_folder_raises_on_failure(monkeypatch):
    monkeypatch.setattr(m, "db_conn", _failing_db_conn("simulated"))
    with pytest.raises(RuntimeError, match="simulated"):
        m.db_delete_folder("SomeFolder")


def test_db_rename_folder_raises_on_failure(monkeypatch):
    monkeypatch.setattr(m, "db_conn", _failing_db_conn("simulated"))
    with pytest.raises(RuntimeError, match="simulated"):
        m.db_rename_folder("Old", "New")


def test_db_mark_document_deleting_raises_on_failure(monkeypatch):
    monkeypatch.setattr(m, "db_conn", _failing_db_conn("simulated"))
    with pytest.raises(RuntimeError, match="simulated"):
        m.db_mark_document_deleting("d1")


def test_db_delete_document_rows_raises_on_failure(monkeypatch):
    monkeypatch.setattr(m, "db_conn", _failing_db_conn("simulated"))
    with pytest.raises(RuntimeError, match="simulated"):
        m.db_delete_document_rows("d1")


def test_init_db_no_longer_swallows_failures(monkeypatch):
    """This used to catch everything and log a warning, letting startup()
    carry on believing it was ready while serving an EMPTY knowledge base."""
    monkeypatch.setattr(m, "db_conn", _failing_db_conn("schema boom"))
    with pytest.raises(RuntimeError, match="schema boom"):
        m.init_db()


# ── create_folder / delete_folder: DB first, memory only after success ─────

async def test_create_folder_does_not_touch_registry_on_db_failure(monkeypatch):
    monkeypatch.setattr(m, "folders_registry", set())
    monkeypatch.setattr(m, "db_save_folder", lambda name: (_ for _ in ()).throw(RuntimeError("db down")))

    with pytest.raises(RuntimeError):
        await m.create_folder({"name": "atomtest-NewFolder"})

    assert "atomtest-NewFolder" not in m.folders_registry


async def test_delete_folder_does_not_touch_registry_on_db_failure(monkeypatch):
    monkeypatch.setattr(m, "folders_registry", {"atomtest-Existing"})
    monkeypatch.setattr(m, "db_delete_folder", lambda name: (_ for _ in ()).throw(RuntimeError("db down")))

    with pytest.raises(RuntimeError):
        await m.delete_folder("atomtest-Existing")

    assert "atomtest-Existing" in m.folders_registry


# ── update_document_folder: durable write first; Qdrant is best-effort ─────

async def test_update_document_folder_does_not_touch_memory_on_db_failure(monkeypatch):
    doc = {"doc_id": "d1", "folder": "Old", "folder_synced": True, "status": "active"}
    monkeypatch.setitem(m.documents_registry, "d1", doc)
    monkeypatch.setattr(m, "db_update_document_folder", lambda doc_id, folder: (_ for _ in ()).throw(RuntimeError("db down")))

    with pytest.raises(RuntimeError):
        await m.update_document_folder("d1", {"folder": "New"})

    assert doc["folder"] == "Old"


async def test_update_document_folder_succeeds_despite_qdrant_failure(monkeypatch):
    """Postgres is the source of truth for folder membership — a Qdrant
    hiccup must not fail a request that already durably succeeded, only
    leave folder_synced=False for later reconciliation."""
    doc = {"doc_id": "d1", "folder": "Old", "folder_synced": True, "status": "active"}
    monkeypatch.setitem(m.documents_registry, "d1", doc)
    monkeypatch.setattr(m, "db_update_document_folder", lambda doc_id, folder: None)
    monkeypatch.setattr(m, "db_save_folder", lambda name: None)
    monkeypatch.setattr(m, "vector_store", _FakeVectorStore(_FakeQdrantClient(fail_always=True)))

    result = await m.update_document_folder("d1", {"folder": "New"})

    assert result == {"doc_id": "d1", "folder": "New", "qdrant_synced": False}
    assert doc["folder"] == "New"
    assert doc["folder_synced"] is False


async def test_update_document_folder_marks_synced_on_qdrant_success(monkeypatch):
    doc = {"doc_id": "d1", "folder": "Old", "folder_synced": True, "status": "active"}
    monkeypatch.setitem(m.documents_registry, "d1", doc)
    monkeypatch.setattr(m, "db_update_document_folder", lambda doc_id, folder: None)
    monkeypatch.setattr(m, "db_save_folder", lambda name: None)
    synced_calls = []
    monkeypatch.setattr(m, "db_mark_folder_synced", lambda doc_id, synced: synced_calls.append((doc_id, synced)))
    monkeypatch.setattr(m, "vector_store", _FakeVectorStore(_FakeQdrantClient()))

    result = await m.update_document_folder("d1", {"folder": "New"})

    assert result["qdrant_synced"] is True
    assert doc["folder_synced"] is True
    assert synced_calls == [("d1", True)]


async def test_concurrent_folder_moves_cannot_publish_older_qdrant_value(monkeypatch):
    """Force A to pause in its DB thread while B tries the same mutation.

    Without the mutation lock B commits/syncs first and A then overwrites
    memory/Qdrant with its stale value.  With serialization B cannot even
    enter its DB step until A has completed the whole cross-store sequence.
    """
    doc = {"doc_id": "d1", "folder": "Old", "folder_synced": True, "status": "active"}
    monkeypatch.setattr(m, "documents_registry", {"d1": doc})
    monkeypatch.setattr(m, "folders_registry", set())
    a_in_db = asyncio.Event()
    release_a = asyncio.Event()
    db_calls = []

    def db_move(doc_id, folder):
        db_calls.append(folder)
        if folder == "A":
            loop.call_soon_threadsafe(a_in_db.set)
            # This function is running in asyncio.to_thread().
            asyncio.run_coroutine_threadsafe(release_a.wait(), loop).result(timeout=2)

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(m, "db_update_document_folder", db_move)
    monkeypatch.setattr(m, "db_mark_folder_synced", lambda doc_id, synced: None)
    qdrant = _FakeQdrantClient()
    monkeypatch.setattr(m, "vector_store", _FakeVectorStore(qdrant))

    task_a = asyncio.create_task(m.update_document_folder("d1", {"folder": "A"}))
    await asyncio.wait_for(a_in_db.wait(), timeout=1)
    task_b = asyncio.create_task(m.update_document_folder("d1", {"folder": "B"}))
    await asyncio.sleep(0)
    assert db_calls == ["A"]  # B is stopped at the mutation lock.
    release_a.set()
    await asyncio.gather(task_a, task_b)

    assert db_calls == ["A", "B"]
    assert doc["folder"] == "B"
    assert qdrant.set_payload_calls == [("d1", {"folder": "A"}), ("d1", {"folder": "B"})]


async def test_startup_reconciliation_cannot_mark_newer_folder_synced(monkeypatch):
    """A startup sweep paused in Qdrant must serialize with a normal move."""
    doc = {"doc_id": "d1", "folder": "Old", "folder_synced": False, "status": "active"}
    monkeypatch.setattr(m, "documents_registry", {"d1": doc})
    monkeypatch.setattr(m, "folders_registry", set())
    reconciliation_in_qdrant = asyncio.Event()
    release_reconciliation = asyncio.Event()
    payloads = []
    loop = asyncio.get_running_loop()

    class PausingQdrant(_FakeQdrantClient):
        def set_payload(self, collection_name, payload, points):
            payloads.append(payload["folder"])
            if payload["folder"] == "Old":
                loop.call_soon_threadsafe(reconciliation_in_qdrant.set)
                asyncio.run_coroutine_threadsafe(release_reconciliation.wait(), loop).result(timeout=2)

    monkeypatch.setattr(m, "db_mark_folder_synced", lambda doc_id, synced: None)
    monkeypatch.setattr(m, "db_update_document_folder", lambda doc_id, folder: None)
    monkeypatch.setattr(m, "vector_store", _FakeVectorStore(PausingQdrant()))

    sweep = asyncio.create_task(m._startup_reconciliation_sweep())
    await asyncio.wait_for(reconciliation_in_qdrant.wait(), timeout=1)
    move = asyncio.create_task(m.update_document_folder("d1", {"folder": "New"}))
    await asyncio.sleep(0)
    assert doc["folder"] == "Old"
    release_reconciliation.set()
    await asyncio.gather(sweep, move)

    assert payloads == ["Old", "New"]
    assert doc["folder"] == "New"
    assert doc["folder_synced"] is True


# ── wait_for backstop: bounded even if the Qdrant client itself hangs ──────

async def test_reconcile_folder_sync_backstop_fires_without_hanging(monkeypatch):
    """QDRANT_RECONCILE_TIMEOUT_SECONDS' wait_for must bound this call even
    if the underlying client call never returns on its own — proving the
    reconciliation helper (and thus the metadata mutation lock) can't be
    wedged open forever by a single stuck Qdrant call."""
    doc = {"doc_id": "d1", "folder": "New", "folder_synced": False, "status": "active"}
    monkeypatch.setattr(m, "documents_registry", {"d1": doc})
    monkeypatch.setattr(m, "QDRANT_RECONCILE_TIMEOUT_SECONDS", 0.05)
    hanging = _HangingQdrantClient()
    monkeypatch.setattr(m, "vector_store", _FakeVectorStore(hanging))

    start = time.monotonic()
    await m._reconcile_document_folder_sync("d1")
    elapsed = time.monotonic() - start

    assert elapsed < 1.0  # bounded by the 0.05s backstop, not the client's own 5s hang
    assert doc["folder_synced"] is False  # treated as an ordinary failure — safe to retry later
    hanging.release_event.set()  # let the orphaned thread finish so it doesn't outlive the test


async def test_reconcile_deletion_backstop_fires_without_hanging(monkeypatch, tmp_path):
    doc = {"doc_id": "d1", "filename": "f.pdf", "status": "deleting"}
    monkeypatch.setattr(m, "documents_registry", {"d1": doc})
    monkeypatch.setattr(m, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(m, "QDRANT_RECONCILE_TIMEOUT_SECONDS", 0.05)
    hanging = _HangingQdrantClient()
    monkeypatch.setattr(m, "vector_store", _FakeVectorStore(hanging))

    start = time.monotonic()
    with pytest.raises(HTTPException) as exc_info:
        await m._reconcile_document_deletion("d1")
    elapsed = time.monotonic() - start

    assert elapsed < 1.0
    assert exc_info.value.status_code == 500
    assert doc["status"] == "deleting"  # still durably pending — safe to retry
    hanging.release_event.set()


async def test_update_document_folder_404s_for_document_being_deleted(monkeypatch):
    doc = {"doc_id": "d1", "folder": "Old", "folder_synced": True, "status": "deleting"}
    monkeypatch.setitem(m.documents_registry, "d1", doc)

    with pytest.raises(HTTPException) as exc_info:
        await m.update_document_folder("d1", {"folder": "New"})
    assert exc_info.value.status_code == 404


# ── rename_folder: durable bulk rename first; per-doc Qdrant sync is independent ──

async def test_rename_folder_does_not_touch_memory_on_db_failure(monkeypatch):
    monkeypatch.setattr(m, "folders_registry", {"atomtest-Old"})
    doc = {"doc_id": "d1", "folder": "atomtest-Old", "folder_synced": True}
    monkeypatch.setitem(m.documents_registry, "d1", doc)
    monkeypatch.setattr(m, "db_rename_folder", lambda old, new: (_ for _ in ()).throw(RuntimeError("db down")))

    with pytest.raises(RuntimeError):
        await m.rename_folder("atomtest-Old", {"name": "atomtest-New"})

    assert doc["folder"] == "atomtest-Old"
    assert "atomtest-Old" in m.folders_registry
    assert "atomtest-New" not in m.folders_registry


async def test_rename_folder_continues_past_a_single_qdrant_failure(monkeypatch):
    """The bug this fixes: the old bare loop had no try/except around the
    per-document Qdrant call — one failure raised out of the loop and left
    every document after it in iteration order completely untouched, even
    in memory, and the bulk Postgres rename never even ran. Now each
    document is reconciled independently and the Postgres rename already
    committed before the loop starts."""
    monkeypatch.setattr(m, "folders_registry", {"atomtest-Old"})
    docs = {
        "d1": {"doc_id": "d1", "folder": "atomtest-Old", "folder_synced": True},
        "d2": {"doc_id": "d2", "folder": "atomtest-Old", "folder_synced": True},
        "d3": {"doc_id": "d3", "folder": "atomtest-Old", "folder_synced": True},
    }
    for doc_id, doc in docs.items():
        monkeypatch.setitem(m.documents_registry, doc_id, doc)
    monkeypatch.setattr(m, "db_rename_folder", lambda old, new: None)
    monkeypatch.setattr(m, "db_mark_folder_synced", lambda doc_id, synced: None)
    monkeypatch.setattr(m, "vector_store", _FakeVectorStore(_FakeQdrantClient(fail_doc_ids={"d2"})))

    result = await m.rename_folder("atomtest-Old", {"name": "atomtest-New"})

    assert result["documents_updated"] == 3
    assert result["qdrant_sync_pending"] == ["d2"]
    # ALL three got the durable/in-memory folder change regardless of
    # Qdrant outcome — only d2's sync flag stays pending.
    assert docs["d1"]["folder"] == "atomtest-New" and docs["d1"]["folder_synced"] is True
    assert docs["d2"]["folder"] == "atomtest-New" and docs["d2"]["folder_synced"] is False
    assert docs["d3"]["folder"] == "atomtest-New" and docs["d3"]["folder_synced"] is True


# ── delete_document: durable tombstone before Qdrant/file, idempotent retry ──

async def test_delete_document_marks_deleting_before_touching_qdrant(monkeypatch, tmp_path):
    doc = {"doc_id": "d1", "filename": "f.pdf", "status": "active"}
    monkeypatch.setitem(m.documents_registry, "d1", doc)
    monkeypatch.setattr(m, "UPLOAD_DIR", tmp_path)
    marked = []
    monkeypatch.setattr(m, "db_mark_document_deleting", lambda doc_id: marked.append(doc_id))
    monkeypatch.setattr(m, "vector_store", _FakeVectorStore(_FakeQdrantClient(fail_always=True)))

    with pytest.raises(HTTPException) as exc_info:
        await m.delete_document("d1")

    assert exc_info.value.status_code == 500
    assert marked == ["d1"]  # tombstone durably written even though Qdrant failed right after
    assert doc["status"] == "deleting"
    assert "d1" in m.documents_registry  # kept — visible to retry logic, hidden from list_documents()


async def test_delete_document_second_call_does_not_rewrite_tombstone(monkeypatch, tmp_path):
    doc = {"doc_id": "d1", "filename": "f.pdf", "status": "deleting"}
    monkeypatch.setitem(m.documents_registry, "d1", doc)
    monkeypatch.setattr(m, "UPLOAD_DIR", tmp_path)
    mark_calls = []
    monkeypatch.setattr(m, "db_mark_document_deleting", lambda doc_id: mark_calls.append(doc_id))
    monkeypatch.setattr(m, "db_delete_document_rows", lambda doc_id: None)
    monkeypatch.setattr(m, "vector_store", _FakeVectorStore(_FakeQdrantClient()))

    await m.delete_document("d1")

    assert mark_calls == []  # already durably 'deleting' — no redundant write


async def test_delete_document_retry_completes_after_transient_failure_clears(monkeypatch, tmp_path):
    doc = {"doc_id": "d1", "filename": "f.pdf", "status": "active"}
    monkeypatch.setitem(m.documents_registry, "d1", doc)
    monkeypatch.setattr(m, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(m, "db_mark_document_deleting", lambda doc_id: None)
    monkeypatch.setattr(m, "db_delete_document_rows", lambda doc_id: None)
    monkeypatch.setattr(m, "vector_store", _FakeVectorStore(_FakeQdrantClient()))

    result = await m.delete_document("d1")

    assert result == {"status": "deleted", "doc_id": "d1", "filename": "f.pdf"}
    assert "d1" not in m.documents_registry


async def test_delete_document_unknown_id_404s(monkeypatch):
    with pytest.raises(HTTPException) as exc_info:
        await m.delete_document("does-not-exist")
    assert exc_info.value.status_code == 404


# ── list_documents: hides tombstoned-but-not-yet-finished deletions ────────

async def test_list_documents_hides_pending_deletions(monkeypatch):
    monkeypatch.setattr(m, "documents_registry", {
        "d1": {"doc_id": "d1", "status": "active"},
        "d2": {"doc_id": "d2", "status": "deleting"},
    })
    monkeypatch.setattr(m, "folders_registry", set())

    result = await m.list_documents()

    assert result["total"] == 1
    assert [d["doc_id"] for d in result["documents"]] == ["d1"]


async def test_tombstone_is_logically_absent_everywhere(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "documents_registry", {
        "dead": {"doc_id": "dead", "filename": "dead.pdf", "format": "pdf", "pages": 1,
                 "folder": "OnlyDead", "status": "deleting"},
        "live": {"doc_id": "live", "folder": "Live", "status": "active"},
    })
    monkeypatch.setattr(m, "folders_registry", set())
    monkeypatch.setattr(m, "UPLOAD_DIR", tmp_path)
    (tmp_path / "dead_dead.pdf").write_bytes(b"%PDF-dead")

    with pytest.raises(HTTPException) as page_error:
        await m.get_document_page("dead", 1)
    assert page_error.value.status_code == 404

    with pytest.raises(HTTPException) as pdf_error:
        await m.get_pdf("dead", key=m.API_KEY)
    assert pdf_error.value.status_code == 404

    folders = await m.list_folders()
    assert folders["folders"] == ["Live"]
    assert m._active_chunks([
        {"document_id": "dead", "text": "stale"},
        {"document_id": "live", "text": "current"},
    ]) == [{"document_id": "live", "text": "current"}]

    request = m.QueryRequest(question="compare", document_ids=["live", "dead"])
    with pytest.raises(HTTPException) as scope_error:
        m._validate_query_document_scope(request)
    assert scope_error.value.status_code == 404


# ── startup reconciliation sweep ────────────────────────────────────────────

async def test_startup_sweep_finishes_pending_deletion(monkeypatch, tmp_path):
    doc = {"doc_id": "d1", "filename": "f.pdf", "status": "deleting"}
    monkeypatch.setattr(m, "documents_registry", {"d1": doc})
    monkeypatch.setattr(m, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(m, "db_delete_document_rows", lambda doc_id: None)
    monkeypatch.setattr(m, "vector_store", _FakeVectorStore(_FakeQdrantClient()))

    await m._startup_reconciliation_sweep()

    assert "d1" not in m.documents_registry


async def test_startup_sweep_finishes_pending_folder_sync(monkeypatch):
    doc = {"doc_id": "d1", "folder": "New", "folder_synced": False, "status": "active"}
    monkeypatch.setattr(m, "documents_registry", {"d1": doc})
    monkeypatch.setattr(m, "db_mark_folder_synced", lambda doc_id, synced: None)
    monkeypatch.setattr(m, "vector_store", _FakeVectorStore(_FakeQdrantClient()))

    await m._startup_reconciliation_sweep()

    assert doc["folder_synced"] is True


async def test_startup_sweep_leaves_still_broken_deletion_pending_without_raising(monkeypatch, tmp_path):
    doc = {"doc_id": "d1", "filename": "f.pdf", "status": "deleting"}
    monkeypatch.setattr(m, "documents_registry", {"d1": doc})
    monkeypatch.setattr(m, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(m, "vector_store", _FakeVectorStore(_FakeQdrantClient(fail_always=True)))

    await m._startup_reconciliation_sweep()  # must not raise — best-effort, one-shot

    assert doc["status"] == "deleting"
    assert "d1" in m.documents_registry
