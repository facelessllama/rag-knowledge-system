"""
Tests for api/main.py's get_document_page() (the unified TXT/PDF source
viewer endpoint) and db_save_ingestion() (the atomic file_hashes/folders/
documents/document_pages ingestion save). Postgres/Qdrant/Ollama are all
faked via monkeypatch — no live services required. embedder/reranker/
vector_store/etc. are assigned inside api/main.py's startup() (see
lock.acquire_single_instance_guard()'s docstring for why), which this file
never triggers, so importing api.main here does NOT load any model onto the
GPU or touch a network — the functions under test here don't need any of
those globals.
"""
import pytest
from fastapi import HTTPException

import api.main as m


# ── get_document_page: TXT branch (reads live off disk) ─────────────────────

async def test_get_document_page_txt_reads_file_from_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "UPLOAD_DIR", tmp_path)
    monkeypatch.setitem(m.documents_registry, "txt1", {"format": "txt", "pages": 1})
    (tmp_path / "txt1_notes.txt").write_bytes("Hello   world.\n\nExtra   spacing.".encode("utf-8"))

    result = await m.get_document_page("txt1", 1)

    assert result["format"] == "txt"
    assert result["total_pages"] == 1
    assert result["text"] == "Hello world. Extra spacing."  # normalize_whitespace collapses runs
    assert result["has_ocr"] is False


async def test_get_document_page_txt_rejects_page_other_than_1(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "UPLOAD_DIR", tmp_path)
    monkeypatch.setitem(m.documents_registry, "txt1", {"format": "txt", "pages": 1})
    (tmp_path / "txt1_notes.txt").write_bytes(b"content")

    with pytest.raises(HTTPException) as exc_info:
        await m.get_document_page("txt1", 2)
    assert exc_info.value.status_code == 404


async def test_get_document_page_txt_missing_file_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "UPLOAD_DIR", tmp_path)
    monkeypatch.setitem(m.documents_registry, "txt1", {"format": "txt", "pages": 1})
    # no file written for txt1 in tmp_path

    with pytest.raises(HTTPException) as exc_info:
        await m.get_document_page("txt1", 1)
    assert exc_info.value.status_code == 404


# ── get_document_page: PDF branch (reads document_pages via Postgres) ──────

class _FakeCursor:
    def __init__(self, row):
        self._row = row
        self.executed = []

    def execute(self, query, params):
        self.executed.append((query, params))

    def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, row):
        self._row = row

    def cursor(self):
        return _FakeCursor(self._row)


def _fake_db_conn(row):
    from contextlib import contextmanager

    @contextmanager
    def _conn():
        yield _FakeConn(row)
    return _conn


async def test_get_document_page_pdf_returns_persisted_normalized_text(monkeypatch):
    monkeypatch.setitem(m.documents_registry, "pdf1", {"format": "pdf", "pages": 3})
    monkeypatch.setattr(m, "db_conn", _fake_db_conn(("page two text here", True)))

    result = await m.get_document_page("pdf1", 2)

    assert result == {
        "document_id": "pdf1", "format": "pdf", "page": 2, "total_pages": 3,
        "text": "page two text here", "has_ocr": True,
    }


async def test_get_document_page_pdf_page_out_of_range_404s(monkeypatch):
    monkeypatch.setitem(m.documents_registry, "pdf1", {"format": "pdf", "pages": 3})
    monkeypatch.setattr(m, "db_conn", _fake_db_conn(None))  # should never be reached

    with pytest.raises(HTTPException) as exc_info:
        await m.get_document_page("pdf1", 4)
    assert exc_info.value.status_code == 404


async def test_get_document_page_pdf_not_backfilled_yet_404s_distinctly(monkeypatch):
    """A PDF whose document_pages row is missing (never backfilled) must
    404 with a message pointing at the cause, not a generic error — this is
    exactly the failure mode a silently-swallowed save error used to
    produce for every subsequent source click."""
    monkeypatch.setitem(m.documents_registry, "pdf1", {"format": "pdf", "pages": 3})
    monkeypatch.setattr(m, "db_conn", _fake_db_conn(None))

    with pytest.raises(HTTPException) as exc_info:
        await m.get_document_page("pdf1", 1)
    assert exc_info.value.status_code == 404
    assert "backfill" in exc_info.value.detail.lower()


async def test_get_document_page_unknown_document_404s(monkeypatch):
    with pytest.raises(HTTPException) as exc_info:
        await m.get_document_page("does-not-exist", 1)
    assert exc_info.value.status_code == 404


# ── db_save_ingestion: atomicity across file_hashes/folders/documents/pages ──

class _RecordingCursor:
    def __init__(self, fail_on=None, zero_rowcount_on=None):
        self.calls = []
        self.fail_on = fail_on  # substring of the query to raise on, or None
        self.zero_rowcount_on = zero_rowcount_on  # substring to report rowcount=0 for (ON CONFLICT no-op)
        self.rowcount = 1

    def execute(self, query, params):
        self.calls.append((query.strip().split()[0], query.strip(), params))
        if self.fail_on and self.fail_on in query:
            raise RuntimeError("simulated DB failure")
        self.rowcount = 0 if (self.zero_rowcount_on and self.zero_rowcount_on in query) else 1


class _RecordingConn:
    def __init__(self, fail_on=None, zero_rowcount_on=None):
        self.cursor_obj = _RecordingCursor(fail_on=fail_on, zero_rowcount_on=zero_rowcount_on)

    def cursor(self):
        return self.cursor_obj


def _recording_db_conn(fail_on=None, zero_rowcount_on=None, holder=None):
    from contextlib import contextmanager

    conn = _RecordingConn(fail_on=fail_on, zero_rowcount_on=zero_rowcount_on)
    if holder is not None:
        holder["conn"] = conn

    @contextmanager
    def _conn():
        yield conn
    return _conn


def test_db_save_ingestion_persists_hash_and_pages_for_pdf(monkeypatch):
    holder = {}
    monkeypatch.setattr(m, "db_conn", _recording_db_conn(holder=holder))

    doc = {"doc_id": "d1", "filename": "f.pdf", "pages": 2, "chunks": 5,
           "size_kb": 10.0, "metadata": {}, "folder": "", "format": "pdf"}
    pages = [{"page_num": 1, "text": "one", "has_ocr": False},
              {"page_num": 2, "text": "two", "has_ocr": False}]

    m.db_save_ingestion(doc, pages, file_hash="abc123")

    calls = holder["conn"].cursor_obj.calls
    # file_hashes insert + documents insert + 2 document_pages inserts (no folder set)
    assert len(calls) == 4
    assert "INTO file_hashes" in calls[0][1]
    assert "INTO documents" in calls[1][1]


def test_db_save_ingestion_also_registers_folder_when_set(monkeypatch):
    holder = {}
    monkeypatch.setattr(m, "db_conn", _recording_db_conn(holder=holder))

    doc = {"doc_id": "d1", "filename": "f.pdf", "pages": 1, "chunks": 1,
           "size_kb": 1.0, "metadata": {}, "folder": "Court Cases", "format": "pdf"}
    pages = [{"page_num": 1, "text": "one", "has_ocr": False}]

    m.db_save_ingestion(doc, pages, file_hash="abc123")

    calls = holder["conn"].cursor_obj.calls
    assert any("INTO folders" in c[1] for c in calls)


def test_db_save_ingestion_skips_pages_for_txt(monkeypatch):
    holder = {}
    monkeypatch.setattr(m, "db_conn", _recording_db_conn(holder=holder))

    doc = {"doc_id": "d1", "filename": "f.txt", "pages": 1, "chunks": 3,
           "size_kb": 1.0, "metadata": {}, "folder": "", "format": "txt"}
    pages = [{"page_num": 1, "text": "whole file", "has_ocr": False}]

    m.db_save_ingestion(doc, pages, file_hash="abc123")

    calls = holder["conn"].cursor_obj.calls
    assert len(calls) == 2  # file_hashes insert + documents insert — no document_pages row for TXT


def test_db_save_ingestion_does_not_swallow_failures(monkeypatch):
    """The old db_save_document/db_save_pages caught everything and only
    logged a warning — /upload then reported "indexed" for a document whose
    page text silently never got saved, and every future source click
    404'd. This must propagate so the caller can fail the whole upload."""
    monkeypatch.setattr(m, "db_conn", _recording_db_conn(fail_on="INSERT INTO document_pages"))

    doc = {"doc_id": "d1", "filename": "f.pdf", "pages": 1, "chunks": 1,
           "size_kb": 1.0, "metadata": {}, "folder": "", "format": "pdf"}
    pages = [{"page_num": 1, "text": "text", "has_ocr": False}]

    with pytest.raises(RuntimeError, match="simulated DB failure"):
        m.db_save_ingestion(doc, pages, file_hash="abc123")


def test_db_save_ingestion_raises_on_concurrent_file_hash_race(monkeypatch):
    """Two truly simultaneous uploads of the same content can both pass the
    caller's in-memory file_hash check before either commits. Without this,
    ON CONFLICT DO NOTHING would let the second one's file_hashes insert
    silently no-op while its own documents/document_pages rows (keyed on
    its own distinct doc_id) commit anyway — two live documents for
    identical content. rowcount == 0 after the insert must force a raise
    instead, so the whole transaction (including this attempt's documents/
    document_pages inserts) rolls back."""
    monkeypatch.setattr(m, "db_conn", _recording_db_conn(zero_rowcount_on="INTO file_hashes"))

    doc = {"doc_id": "d1", "filename": "f.pdf", "pages": 1, "chunks": 1,
           "size_kb": 1.0, "metadata": {}, "folder": "", "format": "pdf"}
    pages = [{"page_num": 1, "text": "text", "has_ocr": False}]

    with pytest.raises(m.DuplicateFileHashError):
        m.db_save_ingestion(doc, pages, file_hash="abc123")


def test_db_save_ingestion_failure_rolls_back_the_file_hash_too():
    """The core atomicity fix: file_hashes, folders, documents and
    document_pages are one transaction now. A failure partway through (here,
    simulated on the document_pages insert) must mean file_hashes never
    actually committed either — verified against a real sqlite-style
    transactional fake, not just call-order bookkeeping, so a rollback that
    silently didn't happen would be caught."""
    from contextlib import contextmanager

    class _TransactionalFakeConn:
        """Mimics psycopg2's commit-on-success / rollback-on-exception
        semantics against a plain in-memory dict, well enough to prove the
        hash row doesn't survive a failed transaction."""
        def __init__(self, committed_store, fail_on):
            self._committed_store = committed_store
            self._pending = dict(committed_store)
            self._fail_on = fail_on
            self.rowcount = 1

        def cursor(self):
            return self

        def execute(self, query, params):
            if self._fail_on in query:
                raise RuntimeError("simulated DB failure")
            if "INTO file_hashes" in query:
                self._pending[params[0]] = params[1]  # hash -> doc_id
                self.rowcount = 1

        def commit(self):
            self._committed_store.update(self._pending)

        def rollback(self):
            pass  # _pending is simply discarded — never merged into committed_store

        def close(self):
            pass

    committed = {}

    @contextmanager
    def fake_db_conn():
        conn = _TransactionalFakeConn(committed, fail_on="INTO document_pages")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    import api.main as m2
    orig = m2.db_conn
    m2.db_conn = fake_db_conn
    try:
        doc = {"doc_id": "d1", "filename": "f.pdf", "pages": 1, "chunks": 1,
               "size_kb": 1.0, "metadata": {}, "folder": "", "format": "pdf"}
        pages = [{"page_num": 1, "text": "text", "has_ocr": False}]
        with pytest.raises(RuntimeError):
            m2.db_save_ingestion(doc, pages, file_hash="abc123")
    finally:
        m2.db_conn = orig

    assert "abc123" not in committed  # the hash never survived the rollback
