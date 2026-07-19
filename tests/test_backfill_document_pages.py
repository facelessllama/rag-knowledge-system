"""
Tests for scripts/backfill_document_pages.py — the self-verifying
comparison logic and the reindex ordering safety property (embed+upsert the
fresh chunk set before deleting the old one, so a failure never leaves a
document with zero points in Qdrant).

Qdrant, Postgres and the embedder are all faked — no real infra/GPU needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest

from backfill_document_pages import (
    compare_chunks, verify_offset_invariant, reindex_document, scroll_all_points,
)
from ingestion.chunker import TextChunk
from vector_db.qdrant_client import point_id_for_chunk


def _chunk(chunk_id, page_num=1, chunk_index=0, char_start=0, char_end=10,
           text="0123456789", has_ocr=False, document_id="doc1"):
    return TextChunk(
        chunk_id=chunk_id, text=text, page_num=page_num, chunk_index=chunk_index,
        char_count=len(text), has_ocr=has_ocr, document_id=document_id,
        char_start=char_start, char_end=char_end,
    )


def _old_point(chunk_id, point_id=None, **payload_overrides):
    # Defaults to the CURRENT deterministic ID for this chunk_id — i.e. "this
    # point has already been migrated onto point_id_for_chunk()" — so tests
    # that aren't specifically about point-ID migration get an old_points
    # fixture that doesn't itself trigger a legacy_point_ids mismatch. Pass
    # an explicit mismatched point_id (e.g. "pid-1") to simulate a
    # pre-migration random-uuid4() point instead.
    if point_id is None:
        point_id = point_id_for_chunk(chunk_id)
    payload = {
        "chunk_id": chunk_id, "page_num": 1, "chunk_index": 0,
        "char_start": 0, "char_end": 10, "text": "0123456789", "has_ocr": False,
    }
    payload.update(payload_overrides)
    return {"point_id": point_id, "payload": payload}


# ── compare_chunks ───────────────────────────────────────────────────────────

def test_compare_chunks_identical_is_ok():
    old_points = {"c1": [_old_point("c1")]}
    new_chunks = [_chunk("c1")]
    ok, report = compare_chunks(old_points, invalid_point_ids=[], new_chunks=new_chunks)
    assert ok
    assert not report["missing"] and not report["extra"] and not report["mismatched"] and not report["duplicates"]


def test_compare_chunks_detects_missing_chunk():
    # existed in Qdrant before, current chunker no longer produces it
    old_points = {"c1": [_old_point("c1")], "c2": [_old_point("c2")]}
    new_chunks = [_chunk("c1")]
    ok, report = compare_chunks(old_points, invalid_point_ids=[], new_chunks=new_chunks)
    assert not ok
    assert report["missing"] == {"c2"}


def test_compare_chunks_detects_extra_chunk():
    # chunker produces a chunk_id that was never in Qdrant
    old_points = {"c1": [_old_point("c1")]}
    new_chunks = [_chunk("c1"), _chunk("c2")]
    ok, report = compare_chunks(old_points, invalid_point_ids=[], new_chunks=new_chunks)
    assert not ok
    assert report["extra"] == {"c2"}


def test_compare_chunks_detects_field_mismatch():
    old_points = {"c1": [_old_point("c1", char_end=10)]}
    new_chunks = [_chunk("c1", char_end=999)]  # offsets drifted
    ok, report = compare_chunks(old_points, invalid_point_ids=[], new_chunks=new_chunks)
    assert not ok
    assert any(f == "char_end" for _, f, _, _ in report["mismatched"])


def test_compare_chunks_detects_duplicate_point_for_same_chunk_id():
    # Two live Qdrant points sharing a chunk_id — a botched prior write.
    # Must fail verification even though every field on both matches the
    # fresh chunk, precisely so this can't be silently collapsed away.
    old_points = {"c1": [_old_point("c1", point_id="pid-A"), _old_point("c1", point_id="pid-B")]}
    new_chunks = [_chunk("c1")]
    ok, report = compare_chunks(old_points, invalid_point_ids=[], new_chunks=new_chunks)
    assert not ok
    assert "c1" in report["duplicates"]
    assert set(report["duplicates"]["c1"]) == {"pid-A", "pid-B"}


def test_compare_chunks_flags_legacy_random_uuid_point_id_as_mismatch():
    """A point written before point_id_for_chunk() existed carries a random
    uuid4() ID unrelated to its chunk_id. Content/offsets can match the
    current chunker perfectly — the only thing wrong is the ID itself — so
    this must be caught independently of the field-by-field comparison,
    otherwise the document is reported 'ok' and never gets migrated onto
    the deterministic scheme: a later plain re-upsert of this same
    unchanged chunk would then create a second point at the new ID rather
    than overwriting this one."""
    old_points = {"c1": [_old_point("c1", point_id="legacy-random-uuid")]}
    new_chunks = [_chunk("c1")]  # text/offsets identical — only the point ID is stale
    ok, report = compare_chunks(old_points, invalid_point_ids=[], new_chunks=new_chunks)
    assert not ok
    assert report["legacy_point_ids"] == {"c1": "legacy-random-uuid"}
    assert not report["mismatched"]  # confirms this isn't being caught via field comparison


def test_compare_chunks_accepts_point_id_already_on_the_deterministic_scheme():
    old_points = {"c1": [_old_point("c1", point_id=point_id_for_chunk("c1"))]}
    new_chunks = [_chunk("c1")]
    ok, report = compare_chunks(old_points, invalid_point_ids=[], new_chunks=new_chunks)
    assert ok
    assert report["legacy_point_ids"] == {}


def test_compare_chunks_flags_invalid_chunk_id_less_points_as_mismatch():
    # A point matching this document but with no chunk_id at all (malformed
    # write) used to be silently skipped by scroll_all_points and never
    # surfaced here — meaning it would never be caught, never be deleted on
    # reindex, and would survive every backfill run forever.
    old_points = {"c1": [_old_point("c1")]}
    new_chunks = [_chunk("c1")]
    ok, report = compare_chunks(old_points, invalid_point_ids=["orphan-pid"], new_chunks=new_chunks)
    assert not ok
    assert report["invalid_point_ids"] == ["orphan-pid"]


# ── scroll_all_points ────────────────────────────────────────────────────────

class _FakeQdrantScrollPoint:
    def __init__(self, point_id, payload):
        self.id = point_id
        self.payload = payload


class _FakeScrollClient:
    """One batch, no pagination needed for these tests."""
    def __init__(self, points):
        self._points = points

    def scroll(self, collection_name, scroll_filter, limit, offset, with_payload, with_vectors):
        return self._points, None


def test_scroll_all_points_separates_chunk_id_less_points_as_invalid():
    points = [
        _FakeQdrantScrollPoint("pid-1", {"chunk_id": "c1", "text": "a"}),
        _FakeQdrantScrollPoint("pid-orphan", {"text": "no chunk_id here at all"}),
    ]
    client = _FakeScrollClient(points)

    by_chunk_id, invalid_point_ids = scroll_all_points(client, "knowledge_base", "doc1")

    assert set(by_chunk_id.keys()) == {"c1"}
    assert invalid_point_ids == ["pid-orphan"]


# ── verify_offset_invariant ──────────────────────────────────────────────────

def test_verify_offset_invariant_passes_when_slice_matches():
    normalized_pages = {1: "0123456789"}
    chunks = [_chunk("c1", page_num=1, char_start=2, char_end=5, text="234")]
    assert verify_offset_invariant(chunks, normalized_pages) == []


def test_verify_offset_invariant_fails_on_drifted_offsets():
    normalized_pages = {1: "0123456789"}
    chunks = [_chunk("c1", page_num=1, char_start=2, char_end=5, text="WRONG")]
    failures = verify_offset_invariant(chunks, normalized_pages)
    assert len(failures) == 1
    assert failures[0][0] == "c1"


def test_verify_offset_invariant_flags_missing_offsets():
    chunk = _chunk("c1")
    chunk.char_start = None
    failures = verify_offset_invariant([chunk], {1: "0123456789"})
    assert failures == [("c1", "no offsets")]


# ── reindex_document ordering (the critical safety property) ────────────────

class _FakeVectorStore:
    def __init__(self, collection="knowledge_base"):
        self.collection = collection
        self.calls = []  # ordered list of "upsert" / "delete" for assertion
        self.client = self

    def upsert_chunks(self, chunks, vectors):
        self.calls.append("upsert")

    def delete(self, collection_name, points_selector):
        self.calls.append("delete")
        self.deleted_selector = points_selector


class _FakeEmbedder:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail

    def embed_batch(self, texts):
        if self.should_fail:
            raise RuntimeError("GPU out of memory (simulated)")
        return [[0.0, 0.0, 0.0] for _ in texts]


def _chunk_context_text(chunk):
    return chunk.text


def test_reindex_document_upserts_before_deleting_old_points():
    """The core ordering fix: new points must be written before old ones are
    removed, so a mid-way failure never leaves the document with zero
    points. Asserted here via call order, not just "both happened"."""
    vs = _FakeVectorStore()
    embedder = _FakeEmbedder()
    parsed = type("Parsed", (), {"total_pages": 1})()
    chunks = [_chunk("c1")]
    old_point_ids = ["pid-1"]

    reindex_document(vs, embedder, _chunk_context_text, "doc1", "f.pdf", "folder", parsed, chunks, old_point_ids)

    assert vs.calls == ["upsert", "delete"]
    assert vs.deleted_selector == old_point_ids  # deletes by exact old point ID, not a document_id filter


def test_reindex_document_leaves_old_points_untouched_if_embedding_fails():
    """If embed_batch blows up, delete() must never be called — the
    document keeps its old (stale but present) points rather than ending up
    with none at all."""
    vs = _FakeVectorStore()
    embedder = _FakeEmbedder(should_fail=True)
    parsed = type("Parsed", (), {"total_pages": 1})()
    chunks = [_chunk("c1")]
    old_point_ids = ["pid-1"]

    with pytest.raises(RuntimeError):
        reindex_document(vs, embedder, _chunk_context_text, "doc1", "f.pdf", "folder", parsed, chunks, old_point_ids)

    assert vs.calls == []  # neither upsert nor delete happened


def test_reindex_document_leaves_old_points_untouched_if_upsert_fails():
    """Same guarantee if the Qdrant write itself fails after a successful
    embed — old points must still be intact."""
    class _FailingUpsertStore(_FakeVectorStore):
        def upsert_chunks(self, chunks, vectors):
            self.calls.append("upsert")
            raise ConnectionError("Qdrant unreachable (simulated)")

    vs = _FailingUpsertStore()
    embedder = _FakeEmbedder()
    parsed = type("Parsed", (), {"total_pages": 1})()
    chunks = [_chunk("c1")]
    old_point_ids = ["pid-1"]

    with pytest.raises(ConnectionError):
        reindex_document(vs, embedder, _chunk_context_text, "doc1", "f.pdf", "folder", parsed, chunks, old_point_ids)

    assert vs.calls == ["upsert"]  # delete never reached


def test_reindex_document_never_deletes_a_point_id_a_fresh_chunk_reuses():
    """Point IDs are deterministic now (vector_db.qdrant_client.
    point_id_for_chunk, derived from chunk_id) — a chunk_id whose boundaries
    didn't move reindexes to the SAME point ID it already had in Qdrant.
    old_point_ids (collected BEFORE upsert_chunks runs above) can therefore
    legitimately include an ID that the upsert just (re)wrote. Deleting it
    unconditionally afterward would delete the point this very call just
    wrote — the exact "briefly lose an unchanged chunk" bug this filtering
    exists to prevent."""
    from vector_db.qdrant_client import point_id_for_chunk

    vs = _FakeVectorStore()
    embedder = _FakeEmbedder()
    parsed = type("Parsed", (), {"total_pages": 1})()
    chunks = [_chunk("c1"), _chunk("c2", chunk_index=1)]
    fresh_id_for_c1 = point_id_for_chunk("c1")  # c1's boundaries are unchanged -> same ID as before
    stale_id = "pid-stale"  # belonged to a chunk_id the new chunk set no longer produces
    old_point_ids = [fresh_id_for_c1, stale_id]

    reindex_document(vs, embedder, _chunk_context_text, "doc1", "f.pdf", "folder", parsed, chunks, old_point_ids)

    assert vs.calls == ["upsert", "delete"]
    assert vs.deleted_selector == [stale_id]  # NOT fresh_id_for_c1 — that one was just rewritten


def test_reindex_document_skips_delete_when_every_old_point_id_survives():
    """If every old point ID matches a freshly-upserted chunk's deterministic
    ID (nothing actually changed), there is nothing stale to delete —
    delete() must not be called at all, not called with an empty selector."""
    from vector_db.qdrant_client import point_id_for_chunk

    vs = _FakeVectorStore()
    embedder = _FakeEmbedder()
    parsed = type("Parsed", (), {"total_pages": 1})()
    chunks = [_chunk("c1")]
    old_point_ids = [point_id_for_chunk("c1")]

    reindex_document(vs, embedder, _chunk_context_text, "doc1", "f.pdf", "folder", parsed, chunks, old_point_ids)

    assert vs.calls == ["upsert"]  # delete never called
