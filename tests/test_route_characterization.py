"""
Characterization tests for api/main.py's route wiring and auth boundaries —
written BEFORE splitting main.py into api/db.py, api/reconciliation.py,
api/health.py, api/documents.py, api/query.py, api/upload.py (see the
refactor plan in progress). Their whole purpose is to catch a router-wiring
mistake that silently drops a route or changes its auth requirement during
that split — not to be an exhaustive functional test of health-check logic
or the ingestion pipeline, which are out of scope for this pass.

Auth is checked by hitting each endpoint through a real TestClient and
asserting the actual status code (black-box behavior) rather than by
inspecting route.dependencies — that stays correct even if the internal
FastAPI/router wiring changes completely, which is exactly what the
refactor is going to do.

/health, /health/live, /health/ready, /models, and /upload had ZERO
TestClient coverage anywhere in this suite before this file.
"""
import io

import pytest
from fastapi.testclient import TestClient

import api.main as m

TEST_API_KEY = "test-secret-key"


async def _unreachable_startup():
    raise AssertionError("startup() must not run for a bare TestClient(app) request")


@pytest.fixture
def keyed_client(monkeypatch):
    """API_KEY set — proves protected routes actually reject an
    unauthenticated request, not just that the route exists."""
    monkeypatch.setattr(m, "startup", _unreachable_startup)
    monkeypatch.setattr(m, "API_KEY", TEST_API_KEY)
    test_client = TestClient(m.app)
    try:
        yield test_client
    finally:
        test_client.close()
        m.app.dependency_overrides.clear()


# ── Auth boundaries ───────────────────────────────────────────────────────

def test_health_endpoints_reachable_without_api_key(keyed_client):
    """/health, /health/live, /health/ready must never require an API key —
    orchestrators/load balancers poll these with no credentials. Whatever
    status they return based on dependency health is fine; 401 specifically
    is not."""
    for path in ("/health", "/health/live", "/health/ready"):
        response = keyed_client.get(path)
        assert response.status_code != 401, f"{path} incorrectly required an API key"


def test_models_requires_api_key(keyed_client):
    assert keyed_client.get("/models").status_code == 401


def test_documents_requires_api_key(keyed_client):
    assert keyed_client.get("/documents").status_code == 401


def test_folders_requires_api_key(keyed_client):
    assert keyed_client.get("/folders").status_code == 401


def test_pdf_route_uses_its_own_manual_auth_not_the_shared_dependency(keyed_client):
    """/pdf/{doc_id} is registered directly on `app`, not `protected` — it
    does its own header-or-query-param check inside the handler, which must
    survive the refactor exactly (see documents.py's public_router in the
    split plan)."""
    assert keyed_client.get("/pdf/some-doc-id").status_code == 401
    assert keyed_client.get("/pdf/some-doc-id?key=wrong").status_code == 401

    # Correct key clears auth — 404 (no such document) proves it got PAST
    # the auth check, not that the document happens to exist.
    assert keyed_client.get(f"/pdf/some-doc-id?key={TEST_API_KEY}").status_code == 404
    assert keyed_client.get(
        "/pdf/some-doc-id", headers={"X-API-Key": TEST_API_KEY}
    ).status_code == 404


# ── /models shape ─────────────────────────────────────────────────────────

class _FakeGeneratorForModels:
    model = "fake-model"
    deepseek = None


def test_models_endpoint_shape_when_authenticated(keyed_client, monkeypatch):
    """No real Ollama reachable in this environment — list_models() catches
    that and falls back to a deterministic shape built from `generator`
    alone, which is what this test actually exercises."""
    monkeypatch.setattr(m, "generator", _FakeGeneratorForModels())
    monkeypatch.setattr(m, "CLOUD_GENERATOR_ENABLED", False)

    response = keyed_client.get("/models", headers={"X-API-Key": TEST_API_KEY})
    assert response.status_code == 200
    body = response.json()
    assert body["current"] == "fake-model"
    assert body["cloud"] == {"enabled_by_admin": False, "available": False, "model": None}


# ── /upload end-to-end with faked heavy services ───────────────────────────

class _FakeEmbedder:
    model_name = "fake-embedder"

    def embed_batch(self, texts, batch_size=32):
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


class _FakeVectorStore:
    def __init__(self):
        self.upserted = []

    def upsert_chunks(self, chunks, vectors):
        self.upserted.append((chunks, vectors))


@pytest.fixture
def upload_client(monkeypatch, tmp_path):
    """Cannot reuse test_query_endpoint.py's app.dependency_overrides
    pattern here — /upload reaches parser/chunker/embedder/vector_store and
    the db_* functions as plain module globals, not through FastAPI DI, so
    each has to be monkeypatched directly on `api.main`. Real TxtParser +
    SmartChunker are used as-is (cheap, pure Python, no GPU/network) —
    only the embedder, vector store, and DB write are faked."""
    from ingestion.chunker import SmartChunker
    from ingestion.txt_parser import TxtParser

    monkeypatch.setattr(m, "startup", _unreachable_startup)
    monkeypatch.setattr(m, "API_KEY", "")
    monkeypatch.setattr(m, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(m, "PARSERS_BY_EXT", {"txt": (TxtParser(), "txt")})
    monkeypatch.setattr(m, "chunker", SmartChunker())
    monkeypatch.setattr(m, "embedder", _FakeEmbedder())
    fake_vector_store = _FakeVectorStore()
    monkeypatch.setattr(m, "vector_store", fake_vector_store)
    monkeypatch.setattr(m, "LANGFUSE_ENABLED", False)
    monkeypatch.setattr(m, "documents_registry", {})
    monkeypatch.setattr(m, "file_hashes", {})
    monkeypatch.setattr(m, "folders_registry", set())

    saved_ingestion = {}

    def _fake_db_save_ingestion(doc, pages, file_hash):
        saved_ingestion["doc_meta"] = doc

    monkeypatch.setattr(m, "db_save_ingestion", _fake_db_save_ingestion)

    test_client = TestClient(m.app)
    try:
        yield test_client, fake_vector_store, saved_ingestion
    finally:
        test_client.close()


def test_upload_endpoint_end_to_end_with_faked_heavy_services(upload_client):
    client, fake_vector_store, saved_ingestion = upload_client
    content = (
        b"This is a short test document with enough real sentence text "
        b"for the chunker to produce at least one chunk from it."
    )

    response = client.post(
        "/upload", files={"file": ("test.txt", io.BytesIO(content), "text/plain")}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "indexed"
    assert body["chunks_created"] >= 1
    assert len(fake_vector_store.upserted) == 1
    assert saved_ingestion["doc_meta"]["filename"] == "test.txt"
