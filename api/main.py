"""
RAG Knowledge Base API
"""
import logging
import math
import time
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv
load_dotenv()

from api import documents, health, upload
from api.documents import (
    _active_chunks,
    _get_active_document,
    _validate_query_document_scope,
    create_folder,
    delete_document,
    delete_folder,
    get_document_page,
    get_pdf,
    list_documents,
    list_folders,
    rename_folder,
    update_document_folder,
)
from api.middleware import MaxBodySizeMiddleware
from api.schemas import ChatTurn, QueryRequest, QueryResponse
from fastapi import FastAPI, APIRouter, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import json
import psycopg2
from psycopg2.extras import RealDictCursor

from langfuse import Langfuse

from ingestion.pdf_parser import PDFParser
from ingestion.txt_parser import TxtParser
from ingestion.chunker import SmartChunker, normalize_whitespace
from lock import (
    acquire_single_instance_guard,
    check_single_instance_guard,
    release_single_instance_guard,
    watch_single_instance_guard,
)
from embeddings.embedding_service import EmbeddingService
from vector_db.qdrant_client import VectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rag.retriever import HybridRetriever
from rag.reranker import CrossEncoderReranker, SimpleReranker
from rag.prompt_builder import PromptBuilder
from rag.generator import LLMGenerator, DeepSeekGenerator, GeneratorRouter
from rag.query_expander import QueryExpander

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Auth
API_KEY = os.getenv("API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def require_api_key(key: str = Security(api_key_header)):
    if not API_KEY:
        return  # key not set — auth disabled
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Deliberately just calling the module-level startup()/shutdown()
    # functions defined further down, not inlining their bodies here —
    # tests/test_startup_guard_ordering.py calls `await m.startup()` and
    # `await m.shutdown()` directly (no ASGI lifespan involved) to exercise
    # the single-instance-guard sequencing without a live Postgres/Qdrant/
    # GPU. Referencing them by name here (resolved at call time, since this
    # generator only actually runs once uvicorn starts the app) keeps that
    # working unchanged while replacing the deprecated @app.on_event API.
    await startup()
    try:
        yield
    finally:
        # Without try/finally, an exception or cancellation raised into this
        # generator after yield (e.g. the ASGI server cancelling it on a
        # shutdown signal) would skip shutdown() entirely — verified live:
        # simulating a post-yield failure produced ["startup"] with no
        # "shutdown" call. shutdown() is what releases the Postgres
        # advisory lock and cancels the watchdog task (see lock.py) — the
        # next instance waiting on that lock has no other way to find out
        # this one is gone.
        await shutdown()


app = FastAPI(title="RAG Knowledge Base API", version="1.0.0", lifespan=lifespan)

# Router for all protected endpoints
protected = APIRouter(dependencies=[Depends(require_api_key)])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# require_not_backing_up() (the FastAPI dependency gating /upload,
# /upload-batch, and every folder/document mutation on lock.py's shared
# flock) now lives independently in api/upload.py and api/documents.py —
# each with its own copy, not a shared import, since it's a yield-based
# generator dependency and delegating through a lazy `import api.main as m`
# wrapper wouldn't reliably forward FastAPI's athrow()-on-route-exception
# into the wrapped generator's own `finally` (see api/documents.py's
# _require_not_backing_up docstring for the full reasoning).

# Minimum cross-encoder rerank score (raw logit, NOT 0-1 cosine similarity —
# ms-marco-MiniLM-L-6-v2 outputs unbounded relevance logits) to attempt an
# answer. Below this, the knowledge base is considered to have no relevant
# content. Checked AFTER reranking, not on the raw hybrid/RRF retrieval
# score — RRF is a rank-fusion formula (Qdrant docs: reciprocal-rank sum,
# not a similarity measure) and gets further boosted by retriever.py's
# multi-query merge logic, so it was never on an interpretable 0-1 scale to
# begin with.
#
# Recalibrated after chunk_context_text() (title-prefixed passages) was
# wired into reranker.rerank() — that change pushed every should-answer
# type's score up together, not just case_summary's, so a single global
# threshold still suffices; per-type thresholds turned out unnecessary (see
# eval/README.md). Measured on golden_dataset.json + heldout_dataset.json
# combined (220 cases): should-refuse scores topped out at 1.00, should-
# answer scores bottomed out at 4.94 — 3.0 sits at the midpoint of that gap,
# not hugging either edge. Re-run eval/run_eval.py on both datasets after
# changing chunk_size, the reranker model, or the dataset composition.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "3.0"))
MAX_CONCURRENT_QUERIES = int(os.getenv("MAX_CONCURRENT_QUERIES", "3"))
_query_semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)

# Passed straight to VectorStore -> QdrantClient(timeout=...) — the REAL
# bound on how long a single Qdrant REST call can block a thread-pool
# thread. _metadata_mutation_lock's reconciliation helpers below layer an
# asyncio.wait_for() on top of that (see QDRANT_RECONCILE_TIMEOUT_SECONDS),
# but that outer wait is a backstop, not the primary bound — a shorter
# asyncio-level timeout than this one would just abandon the *coroutine*
# waiting on the call while the underlying thread (and its outstanding HTTP
# request) keeps running past it; if the mutation lock is released at that
# point, that orphaned call can still complete later and write a now-stale
# value (a folder that's since been renamed again, say) with nothing left
# holding the lock to stop it. See _reconcile_document_folder_sync()'s and
# _reconcile_document_deletion_locked()'s comments.
QDRANT_REQUEST_TIMEOUT_SECONDS = float(os.getenv("QDRANT_REQUEST_TIMEOUT_SECONDS", "5"))
# Deliberately several times QDRANT_REQUEST_TIMEOUT_SECONDS, not a tighter
# independent budget: sized so the underlying REST call is guaranteed to
# have already finished (successfully or with its own client-level
# TimeoutError) well before this could ever fire. Under normal operation
# this wait_for should never actually time out; it exists only so a bug in
# qdrant_client/httpx itself (the client-level timeout failing to fire)
# can't wedge the metadata-mutation lock open forever instead of just
# failing this one reconciliation attempt.
QDRANT_RECONCILE_TIMEOUT_SECONDS = QDRANT_REQUEST_TIMEOUT_SECONDS * 3


class _ReentrantAsyncLock:
    """Task-reentrant lock for cross-store metadata mutations.

    Folder/delete endpoints call reconciliation helpers while holding this
    lock, and the startup sweep calls those same helpers directly.  A plain
    asyncio.Lock would deadlock on the nested call; task ownership lets both
    entry paths use one lock without leaving an accidentally-unlocked helper.

    One instance is meant to live for exactly one event loop's lifetime — in
    this process, that's the whole run (uvicorn owns a single loop end to
    end). It deliberately does NOT try to detect or repair being reused
    across a different event loop (an earlier version did, purely to let a
    single module-level instance survive pytest-asyncio's fresh loop per
    test): that logic never does anything in production — this app only
    ever has one loop — so it was dead weight on every read of this class
    for a problem only tests have. tests/test_folder_document_atomicity.py's
    `_fresh_metadata_mutation_lock` autouse fixture solves it properly
    instead, by giving every test its own instance up front.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._owner = None
        self._depth = 0

    async def __aenter__(self):
        task = asyncio.current_task()
        if task is self._owner:
            self._depth += 1
            return self
        await self._lock.acquire()
        self._owner = task
        self._depth = 1
        return self

    async def __aexit__(self, exc_type, exc, tb):
        task = asyncio.current_task()
        if task is not self._owner:
            raise RuntimeError("metadata mutation lock released by a non-owner task")
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
            self._lock.release()


# A single worker still serves multiple coroutines concurrently.  Keep every
# folder/delete/reconciliation sequence serialized across its DB, registry,
# disk and Qdrant awaits so an older request cannot publish stale derived
# state after a newer request has committed.
_metadata_mutation_lock = _ReentrantAsyncLock()

# Upload/ingestion limits. Unbounded values here previously meant: the
# whole file read into RAM regardless of size, no cap on how many files one
# batch request could contain, and no cap on PDF page count (so OCR time
# scaled with an attacker/mistake-controlled input). MAX_CONCURRENT_
# INGESTIONS (bounding parse/OCR/embed concurrency) lives in api/upload.py
# now, self-contained — nothing else references it.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_BATCH_FILES = int(os.getenv("MAX_BATCH_FILES", "20"))

# +2MB slack on top of the strict per-file/per-batch byte caps above, for
# multipart framing overhead (boundaries, headers) and the small non-file
# form fields (`folder`) — enforced at the ASGI layer, see
# MaxBodySizeMiddleware's docstring for why MAX_UPLOAD_BYTES alone isn't
# early enough. Every other route gets DEFAULT_MAX_BODY_BYTES; none of them
# have a legitimate reason to receive more than a small JSON body.
_BODY_OVERHEAD_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_BODY_BYTES = 2 * 1024 * 1024
app.add_middleware(
    MaxBodySizeMiddleware,
    limits={
        "/upload": MAX_UPLOAD_BYTES + _BODY_OVERHEAD_BYTES,
        "/upload-batch": MAX_BATCH_FILES * MAX_UPLOAD_BYTES + _BODY_OVERHEAD_BYTES,
    },
    default_limit=DEFAULT_MAX_BODY_BYTES,
)

# parser/txt_parser/PARSERS_BY_EXT (like embedder/reranker/... below) are
# assigned in startup(), not here — see the comment above that block for why.
parser = None
txt_parser = None
PARSERS_BY_EXT = None

# chunker/embedder/vector_store/retriever/reranker/prompt_builder/generator/
# query_expander/langfuse/LANGFUSE_ENABLED are all assigned in startup()
# below, not at module import time as they used to be — embedder and
# reranker in particular load real models onto the GPU, which is expensive
# (memory + load time) to pay before knowing this process is even allowed to
# run. startup() acquires lock.py's acquire_single_instance_guard() as its
# very first action, before constructing any of these, so a second
# process/worker/replica fails fast — without ever loading a model — instead
# of paying the full load cost on every crash-loop respawn only to be
# rejected afterward. See acquire_single_instance_guard()'s docstring for
# why running more than one process against the same in-memory registries is
# unsafe at all.
chunker = None
embedder = None
vector_store = None
retriever = None
reranker = None
prompt_builder = None
generator = None
query_expander = None
langfuse = None
LANGFUSE_ENABLED = False
# Whether the administrator has opted the whole server into the DeepSeek
# cloud generator being selectable at all (ENABLE_CLOUD_GENERATOR env var,
# read once at startup) — the FIRST of GeneratorRouter's two required
# gates; the second is the per-request provider="deepseek" field. Exposed
# by /models so the frontend knows whether to offer the toggle.
CLOUD_GENERATOR_ENABLED = False

# Dependency-provider functions for the /query and /query/stream endpoints'
# heavy services — thin wrappers over the module globals above, wired via
# FastAPI's Depends() instead of the endpoints reading query_expander/
# retriever/reranker/prompt_builder/generator directly. Production behavior
# is unchanged (each provider just returns the same global startup() sets),
# but this makes the services swappable per-request via
# app.dependency_overrides — see tests/fakes.py and
# tests/test_query_endpoint.py, which drive /query through a real TestClient
# (routing, auth, Pydantic validation all actually run) without ever
# constructing EmbeddingService/CrossEncoderReranker/LLMGenerator, so no GPU
# load or Ollama/Qdrant network call happens in that test.
def get_query_expander():
    return query_expander


def get_retriever():
    return retriever


def get_reranker():
    return reranker


def get_prompt_builder():
    return prompt_builder


def get_generator():
    return generator


# api/query.py imports get_query_expander/get_retriever/get_reranker/
# get_prompt_builder/get_generator directly at its own top level (needed as
# Depends(...) default-argument values, evaluated at route-decoration time)
# — so this import must come after all five are defined above, unlike
# api/documents.py's and api/health.py's imports (which only ever reach
# into api.main lazily, inside function bodies, so have no such ordering
# constraint). See api/query.py's own module docstring for the full
# explanation.
from api import query

documents_registry: dict = {}
file_hashes: dict = {}

from contextlib import contextmanager

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://raguser:ragpass@localhost:5432/ragdb")

def get_db():
    return psycopg2.connect(POSTGRES_URL)

@contextmanager
def db_conn():
    conn = get_db()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

folders_registry: set = set()  # persisted folder names

def init_db():
    # No try/except here — deliberately. This used to swallow any failure
    # (schema creation, or the SELECTs that populate documents_registry/
    # file_hashes/folders_registry) into a logged warning and let startup()
    # carry on as if the app were ready, silently serving an EMPTY
    # knowledge base instead of failing loudly. init_db() runs inside
    # startup()'s own try/except now (see api/main.py's startup()), which
    # already treats any exception here as fatal — cancels the watchdog,
    # releases the single-instance guard, and lets the process exit — the
    # same "fail the whole startup, don't limp along on broken state"
    # policy the rest of that function follows.
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS file_hashes (
                hash VARCHAR(32) PRIMARY KEY,
                doc_id UUID NOT NULL,
                filename VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS folders (
                name VARCHAR(255) PRIMARY KEY,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        # Source of truth for document metadata — startup restore reads this
        # table instead of scrolling the entire Qdrant collection.
        # doc_id is a full uuid4() (see upload_document/upload_batch) — a
        # truncated 8-hex-char id (32 bits of entropy) was replaced after
        # the birthday-paradox collision probability at this project's own
        # tested 57k-document scale (see eval/README.md) came out to
        # ~31.5%, which ON CONFLICT DO UPDATE below would resolve by
        # silently overwriting one document's metadata and mixing its
        # Qdrant points with another's.
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
        # documents table may already exist from before `format` was added
        cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS format VARCHAR(10) DEFAULT 'pdf'")
        # status: 'active' | 'deleting' — a durable tombstone written
        # BEFORE delete_document() touches Qdrant or the file on disk, so
        # a crash/restart mid-delete leaves a row this same startup (see
        # _startup_reconciliation_sweep()) or a repeated DELETE call can
        # find and finish, instead of the object either vanishing from
        # Postgres while still orphaned on disk, or silently reappearing
        # after a partial failure. See delete_document()'s docstring.
        cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'active'")
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'documents_status_check'
                      AND conrelid = 'documents'::regclass
                ) THEN
                    ALTER TABLE documents
                    ADD CONSTRAINT documents_status_check
                    CHECK (status IN ('active', 'deleting'));
                END IF;
            END
            $$
        """)
        # folder_synced: whether Qdrant's payload.folder for this
        # document is known to match the `folder` column here. Postgres
        # is the source of truth for folder membership (list/get/delete
        # all read `folder` from here, never from Qdrant) — Qdrant's copy
        # only affects folder-scoped search and is written best-effort,
        # set back to TRUE only once Qdrant actually confirms the write.
        # A rename/move that partially fails leaves this FALSE for the
        # documents Qdrant didn't confirm, so reconciliation later
        # retries exactly those, not all of them. See rename_folder()/
        # update_document_folder()/_reconcile_document_folder_sync().
        cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS folder_synced BOOLEAN NOT NULL DEFAULT TRUE")
        # Canonical source for the document viewer / highlighting: the exact
        # normalized text char_start/char_end (Qdrant chunk payload) are
        # offsets into, persisted once at ingestion — never re-derived by
        # re-running PDFParser/OCR later, since a library or Tesseract
        # config change would silently desync those offsets from what's
        # shown. ON DELETE CASCADE keeps this in lockstep with `documents`
        # without a separate delete step in delete_document().
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_pages (
                document_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE ON UPDATE CASCADE,
                page_num INTEGER NOT NULL,
                normalized_text TEXT NOT NULL,
                has_ocr BOOLEAN DEFAULT FALSE,
                char_count INTEGER DEFAULT 0,
                PRIMARY KEY (document_id, page_num)
            )
        """)
        cur.execute("SELECT hash, doc_id FROM file_hashes")
        for row in cur.fetchall():
            file_hashes[row[0]] = row[1]
        cur.execute("SELECT name FROM folders")
        for row in cur.fetchall():
            folders_registry.add(row[0])
        cur.execute(
            "SELECT doc_id, filename, pages, chunks, size_kb, metadata, folder, format, status, folder_synced "
            "FROM documents"
        )
        for doc_id, filename, pages, chunks, size_kb, metadata, folder, doc_format, status, folder_synced in cur.fetchall():
            documents_registry[doc_id] = {
                "doc_id": doc_id,
                "filename": filename,
                "pages": pages,
                "chunks": chunks,
                "size_kb": size_kb,
                "metadata": metadata or {},
                "folder": folder or "",
                "format": doc_format or "pdf",
                "status": status or "active",
                "folder_synced": True if folder_synced is None else folder_synced,
            }
    logger.info(
        f"DB ready | loaded {len(file_hashes)} hashes, {len(folders_registry)} folders, "
        f"{len(documents_registry)} documents"
    )


class DuplicateFileHashError(Exception):
    """Raised by db_save_ingestion() when another request already committed
    this exact file_hash first — see its docstring's concurrency note."""


def db_save_ingestion(doc: dict, pages: list[dict] | None, file_hash: str):
    """Persists everything a successful upload commits to Postgres —
    file_hashes, folders, documents, and (PDF only — see
    get_document_page()'s TXT branch, which reads straight off disk
    instead) document_pages — in ONE transaction. Does NOT swallow
    failures: a caller that gets an exception here must treat the whole
    upload as failed and roll back Qdrant + the uploaded file, not report
    it as indexed (see upload_document/upload_batch).

    Previously file_hashes was written by a separate, earlier, best-effort
    call before document_pages even existed as a concept — if the later
    document/pages save then failed, the hash row (and the in-memory
    file_hashes entry a caller only updates *after* this function returns
    successfully) never got cleaned up. Every retry of the same file then
    hit the file_hash-seen check and got a false 409 "Already uploaded",
    even though the document, its Qdrant points, and its file on disk were
    all gone — and, if the hash row had already committed to Postgres, this
    survived an API restart too. Folding it into this same transaction
    means a failure here rolls back the hash right along with everything
    else, so the caller's Qdrant/file rollback is the only cleanup left to
    do by hand.

    Concurrency: two truly simultaneous uploads of the same file content can
    both pass the caller's in-memory `file_hash in file_hashes` check before
    either has committed — the in-memory dict only reflects committed state,
    so two in-flight requests never see each other there. Without a check
    here, ON CONFLICT DO NOTHING would let the second one's file_hashes
    insert silently no-op while its documents/document_pages inserts (keyed
    on its own distinct doc_id, not the hash) go on to commit fine — two
    separate documents indexed for identical content, with file_hashes only
    ever pointing at one of them. cursor.rowcount == 0 after the insert
    means someone else's row is already there; raising forces this whole
    transaction to roll back (nothing about this doc_id gets persisted) so
    the caller can treat it as a 409 and clean up the Qdrant points/file
    this now-redundant attempt already wrote, instead of ending up with a
    second live copy of a document that's already in the knowledge base.

    Page persistence being required for PDF: the source viewer 404s
    without it. The same normalize_whitespace() output chunker.py computed
    char_start/char_end against for every chunk — re-deriving this later by
    re-running PDFParser/OCR would risk desyncing from those already-stored
    offsets (see document_pages table comment in init_db())."""
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO file_hashes (hash, doc_id, filename) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
            (file_hash, doc["doc_id"], doc["filename"]),
        )
        if cur.rowcount == 0:
            raise DuplicateFileHashError(
                f"file_hash {file_hash} was already claimed by a concurrent upload"
            )
        if doc.get("folder"):
            cur.execute("INSERT INTO folders (name) VALUES (%s) ON CONFLICT DO NOTHING", (doc["folder"],))
        cur.execute(
            """
            INSERT INTO documents (doc_id, filename, pages, chunks, size_kb, metadata, folder, format)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE SET
                filename = EXCLUDED.filename, pages = EXCLUDED.pages, chunks = EXCLUDED.chunks,
                size_kb = EXCLUDED.size_kb, metadata = EXCLUDED.metadata, folder = EXCLUDED.folder,
                format = EXCLUDED.format
            """,
            (doc["doc_id"], doc["filename"], doc["pages"], doc["chunks"], doc["size_kb"],
             json.dumps(doc.get("metadata", {})), doc.get("folder", ""), doc.get("format", "pdf")),
        )
        if doc.get("format") == "pdf" and pages:
            for p in pages:
                norm = normalize_whitespace(p["text"])
                cur.execute(
                    """
                    INSERT INTO document_pages (document_id, page_num, normalized_text, has_ocr, char_count)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (document_id, page_num) DO UPDATE SET
                        normalized_text = EXCLUDED.normalized_text,
                        has_ocr = EXCLUDED.has_ocr,
                        char_count = EXCLUDED.char_count
                    """,
                    (doc["doc_id"], p["page_num"], norm, p.get("has_ocr", False), len(norm)),
                )


# None of the functions below swallow exceptions (an earlier version of
# each caught Exception and just logged a warning). That made every caller
# unable to tell a durable write from a silently-failed one — the caller
# would go on to mutate documents_registry/folders_registry as if the DB
# write had happened, so a Postgres hiccup left in-memory state that a
# restart (which reloads straight from Postgres) would quietly revert.
# Every caller of these now does the DB write FIRST and only touches memory
# after it returns without raising — see update_document_folder(),
# create_folder(), delete_folder(), rename_folder() below.

def db_update_document_folder(doc_id: str, folder: str):
    with db_conn() as conn:
        cur = conn.cursor()
        if folder:
            # Folder creation and the move are one transaction.  Previously
            # the endpoint committed db_save_folder() first, so a failed
            # document UPDATE returned 500 but leaked an empty folder.
            cur.execute("INSERT INTO folders (name) VALUES (%s) ON CONFLICT DO NOTHING", (folder,))
        cur.execute(
            "UPDATE documents SET folder = %s, folder_synced = FALSE "
            "WHERE doc_id = %s AND status = 'active'", (folder, doc_id)
        )
        if cur.rowcount != 1:
            raise LookupError(f"Active document {doc_id} not found")

def db_save_folder(name: str):
    with db_conn() as conn:
        conn.cursor().execute("INSERT INTO folders (name) VALUES (%s) ON CONFLICT DO NOTHING", (name,))

def db_delete_folder(name: str):
    with db_conn() as conn:
        conn.cursor().execute("DELETE FROM folders WHERE name = %s", (name,))

def db_rename_folder(old_name: str, new_name: str):
    """Renames a folder and marks every document in it as needing a Qdrant
    resync, all in ONE Postgres transaction (db_conn() commits at the end or
    rolls back everything on any failure — see db_conn()'s docstring). This
    is the durable, atomic part of a folder rename; the per-document Qdrant
    payload updates that follow it in rename_folder() are a separate,
    best-effort, individually-retryable step — see folder_synced's comment
    in init_db() and _reconcile_document_folder_sync()."""
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO folders (name) VALUES (%s) ON CONFLICT DO NOTHING", (new_name,))
        cur.execute("DELETE FROM folders WHERE name = %s", (old_name,))
        cur.execute(
            "UPDATE documents SET folder = %s, folder_synced = FALSE WHERE folder = %s", (new_name, old_name)
        )

def db_mark_document_deleting(doc_id: str):
    """The durable tombstone delete_document() writes BEFORE touching
    Qdrant, the file on disk, or documents_registry — see its docstring."""
    with db_conn() as conn:
        conn.cursor().execute("UPDATE documents SET status = 'deleting' WHERE doc_id = %s", (doc_id,))

def db_delete_document_rows(doc_id: str):
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM file_hashes WHERE doc_id = %s", (doc_id,))
        cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))

def db_mark_folder_synced(doc_id: str, synced: bool):
    with db_conn() as conn:
        conn.cursor().execute("UPDATE documents SET folder_synced = %s WHERE doc_id = %s", (synced, doc_id))


async def _reconcile_document_folder_sync(doc_id: str):
    """Retries writing a document's current (Postgres-durable) folder into
    Qdrant's payload if the last attempt didn't confirm success —
    documents_registry[doc_id]["folder_synced"] is False (see folder_synced's
    comment in init_db()). Safe to call repeatedly or on a document that's
    already synced (no-op): Qdrant's set_payload is idempotent on the same
    value, and this only ever records success in Postgres/memory after
    Qdrant itself confirms the write, never before."""
    async with _metadata_mutation_lock:
        doc = documents_registry.get(doc_id)
        if (doc is None or doc.get("status", "active") != "active"
                or doc.get("folder_synced", True)):
            return
        try:
            # wait_for is a backstop, not the timeout — see
            # QDRANT_RECONCILE_TIMEOUT_SECONDS's comment: VectorStore's own
            # client-level timeout (QDRANT_REQUEST_TIMEOUT_SECONDS) is what
            # actually bounds the underlying HTTP call/thread; this is sized
            # strictly larger so it only fires if that mechanism itself
            # fails, and by then the thread is guaranteed to already be done
            # — a set_payload() this stale is never still in flight when the
            # lock below is about to be released.
            await asyncio.wait_for(
                asyncio.to_thread(
                    vector_store.client.set_payload,
                    collection_name=vector_store.collection,
                    payload={"folder": doc["folder"]},
                    points=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))])
                ),
                timeout=QDRANT_RECONCILE_TIMEOUT_SECONDS,
            )
            await asyncio.to_thread(db_mark_folder_synced, doc_id, True)
            doc["folder_synced"] = True
        except Exception as e:
            logger.warning(f"Qdrant folder resync still pending for {doc_id}: {e}")


async def _reconcile_document_deletion(doc_id: str) -> dict:
    """Finishes deleting a document already durably marked status='deleting'
    in Postgres (see delete_document(), which writes that tombstone BEFORE
    calling this). Safe to call repeatedly on the same doc_id — Qdrant's
    delete-by-filter and unlink(missing_ok=True) are both no-ops on content
    that's already gone, so a retry after a partial failure only redoes
    whatever didn't confirm last time, never re-fails on what already
    succeeded. Called both from delete_document() itself (first attempt, and
    any manual retry via calling DELETE again) and from
    _startup_reconciliation_sweep() (automatic retry for anything a crash
    left half-finished)."""
    async with _metadata_mutation_lock:
        return await _reconcile_document_deletion_locked(doc_id)


async def _reconcile_document_deletion_locked(doc_id: str) -> dict:
    """Implementation called with _metadata_mutation_lock held."""
    doc = documents_registry.get(doc_id)
    if doc is None:
        raise HTTPException(404, f"Document {doc_id} not found")
    filename = doc.get("filename", doc_id)
    errors = {}

    try:
        # Same backstop reasoning as _reconcile_document_folder_sync()'s
        # set_payload() call above — delete-by-filter is idempotent, so a
        # late-completing orphaned call here is far less dangerous than a
        # stale set_payload() would be (it can only re-delete something
        # already gone, never resurrect stale data), but there's no reason
        # to skip the same defense-in-depth.
        await asyncio.wait_for(
            asyncio.to_thread(
                vector_store.client.delete,
                collection_name=vector_store.collection,
                points_selector=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))])
            ),
            timeout=QDRANT_RECONCILE_TIMEOUT_SECONDS,
        )
    except Exception as e:
        logger.error(f"Qdrant delete failed for {doc_id}: {e}")
        errors["qdrant"] = str(e)

    try:
        def _delete_files():
            for f in UPLOAD_DIR.glob(f"{doc_id}_*"):
                f.unlink(missing_ok=True)
        await asyncio.to_thread(_delete_files)
    except Exception as e:
        logger.error(f"File delete failed for {doc_id}: {e}")
        errors["file"] = str(e)

    if errors:
        logger.error(f"Document {doc_id} still pending deletion (status='deleting'), failed steps: {list(errors.keys())}")
        raise HTTPException(
            500,
            f"Partial deletion failure for {doc_id}. Failed steps: {list(errors.keys())}. "
            f"Document is durably marked for deletion in Postgres — safe to retry (call DELETE "
            f"again, or it will be retried automatically on next restart)."
        )

    try:
        await asyncio.to_thread(db_delete_document_rows, doc_id)
    except Exception as e:
        logger.error(f"DB row delete failed for {doc_id} after Qdrant/file cleanup succeeded: {e}")
        raise HTTPException(
            500,
            f"Qdrant and file cleanup for {doc_id} succeeded but removing its DB row failed: {e}. Retry."
        )

    for h in [h for h, d in file_hashes.items() if d == doc_id]:
        del file_hashes[h]
    documents_registry.pop(doc_id, None)
    return {"status": "deleted", "doc_id": doc_id, "filename": filename}


_startup_reconciliation_task = None


async def _startup_reconciliation_sweep():
    """One-shot, best-effort pass over state just loaded from Postgres,
    retrying anything a previous crash/restart could have left
    half-finished: documents durably marked status='deleting' (see
    delete_document()) and documents whose folder_synced is False (see
    rename_folder()/update_document_folder()). Fired from startup() as a
    background task (not awaited) — does not block readiness, since Qdrant
    or the filesystem being slow or briefly unavailable here shouldn't hold
    up serving requests that don't touch the affected documents. This is
    the ONLY place a delete/folder-sync retry ever happens without a human
    (or the frontend) triggering it directly — deliberately a single pass at
    startup, not a recurring interval worker; anything it can't finish stays
    durably marked and gets picked up by the next restart or the next
    manual retry (calling DELETE again, or renaming the folder again)."""
    pending_deletions = [doc_id for doc_id, doc in list(documents_registry.items()) if doc.get("status") == "deleting"]
    for doc_id in pending_deletions:
        try:
            await _reconcile_document_deletion(doc_id)
            logger.info(f"Startup reconciliation: finished pending deletion for {doc_id}")
        except HTTPException as e:
            logger.warning(f"Startup reconciliation: deletion for {doc_id} still incomplete: {e.detail}")

    pending_folder_syncs = [doc_id for doc_id, doc in list(documents_registry.items()) if not doc.get("folder_synced", True)]
    for doc_id in pending_folder_syncs:
        await _reconcile_document_folder_sync(doc_id)

    if pending_deletions or pending_folder_syncs:
        logger.info(
            f"Startup reconciliation swept {len(pending_deletions)} pending deletion(s) and "
            f"{len(pending_folder_syncs)} pending folder-sync(s) left over from a previous run."
        )


# Flipped false by the single-instance watchdog's on_failure callback the
# instant it can no longer prove this process holds the Postgres advisory
# lock (see lock.watch_single_instance_guard()) — checked by /health below.
# In practice the watchdog calls os._exit(1) in the very next line after
# flipping this, with no `await` in between, so there is normally no window
# in which a request could actually observe it false; it exists as a
# defense-in-depth belt for that call site, not as the real stop mechanism
# (the process dying is).
_single_instance_healthy = False
_single_instance_watchdog_task = None


async def startup():
    global chunker, embedder, vector_store, retriever, reranker, prompt_builder, \
        generator, query_expander, langfuse, LANGFUSE_ENABLED, parser, txt_parser, PARSERS_BY_EXT, \
        CLOUD_GENERATOR_ENABLED, \
        _single_instance_healthy, _single_instance_watchdog_task, _startup_reconciliation_task

    # Acquired before constructing anything below — in particular before
    # embedder/reranker, which load real models onto the GPU. A second
    # process (`uvicorn --workers 2`, a second replica, a crash-loop respawn
    # of a rejected worker) fails here, immediately, without ever paying that
    # load cost. See acquire_single_instance_guard()'s docstring in lock.py.
    try:
        acquire_single_instance_guard(POSTGRES_URL)
    except RuntimeError as e:
        logger.critical(
            "Refusing to start: %s. documents_registry/file_hashes/"
            "folders_registry are process-local in-memory state loaded once "
            "from Postgres at startup — running a second instance (e.g. "
            "`uvicorn --workers 2`, or two replicas) would silently desync "
            "list/page/pdf/delete depending on which instance a request "
            "lands on. Run with a single worker/replica, or migrate those "
            "registries to read from Postgres at request time before "
            "removing this guard.", e,
        )
        raise

    def _mark_single_instance_unhealthy():
        global _single_instance_healthy
        _single_instance_healthy = False

    _single_instance_watchdog_task = None
    try:
        watchdog_interval = float(os.getenv("SINGLE_INSTANCE_WATCHDOG_INTERVAL_SECONDS", "5"))
        watchdog_timeout = float(os.getenv("SINGLE_INSTANCE_WATCHDOG_TIMEOUT_SECONDS", "5"))
        if not math.isfinite(watchdog_interval) or watchdog_interval <= 0:
            raise ValueError("SINGLE_INSTANCE_WATCHDOG_INTERVAL_SECONDS must be a finite number greater than zero")
        if not math.isfinite(watchdog_timeout) or watchdog_timeout <= 0:
            raise ValueError("SINGLE_INSTANCE_WATCHDOG_TIMEOUT_SECONDS must be a finite number greater than zero")

        parser = PDFParser(
            ocr_language=os.getenv("PDF_OCR_LANGUAGE", "eng"),
            max_pages=int(os.getenv("MAX_PDF_PAGES", "500")),
            ocr_timeout_seconds=int(os.getenv("OCR_TIMEOUT_SECONDS", "60")),
            max_total_ocr_seconds=int(os.getenv("MAX_TOTAL_OCR_SECONDS", "600")),
        )
        txt_parser = TxtParser()
        PARSERS_BY_EXT = {"pdf": (parser, "pdf"), "txt": (txt_parser, "txt")}

        chunker = SmartChunker(
            chunk_size=int(os.getenv("MAX_CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50"))
        )
        embedder = EmbeddingService(model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
        vector_store = VectorStore(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection=os.getenv("QDRANT_COLLECTION", "knowledge_base"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=QDRANT_REQUEST_TIMEOUT_SECONDS,
        )
        retriever = HybridRetriever(embedder, vector_store)
        try:
            reranker = CrossEncoderReranker(model_name=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        except Exception as e:
            logger.warning(f"CrossEncoderReranker failed to load ({e}), falling back to SimpleReranker")
            reranker = SimpleReranker()
        prompt_builder = PromptBuilder()
        local_generator = LLMGenerator(
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11435"),
            model=os.getenv("LLM_MODEL", "qwen2.5:7b"),
            temperature=float(os.getenv("TEMPERATURE", "0.1"))
        )
        # CLOUD_GENERATOR_ENABLED is the administrator's opt-in
        # (ENABLE_CLOUD_GENERATOR=true) — the product default is local-only
        # regardless of whether a DEEPSEEK_API_KEY happens to be present in
        # the environment, so simply having a key configured (e.g. left over
        # from eval work) can never make the cloud provider selectable on
        # its own. A key is ALSO required once the administrator does opt
        # in; missing either one means GeneratorRouter refuses
        # provider="deepseek" with a clear error rather than silently
        # running local instead — see rag/generator.py's GeneratorRouter.
        CLOUD_GENERATOR_ENABLED = os.getenv("ENABLE_CLOUD_GENERATOR", "false").lower() == "true"
        deepseek_generator = None
        if CLOUD_GENERATOR_ENABLED:
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            if deepseek_api_key:
                deepseek_generator = DeepSeekGenerator(
                    api_key=deepseek_api_key,
                    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
                    temperature=float(os.getenv("TEMPERATURE", "0.1")),
                )
                logger.info(f"Cloud generator ENABLED | model={deepseek_generator.model}")
            else:
                logger.warning(
                    "ENABLE_CLOUD_GENERATOR=true but DEEPSEEK_API_KEY is not set — "
                    "provider='deepseek' requests will be rejected with a clear error, "
                    "not silently served by the local model."
                )
        generator = GeneratorRouter(
            local=local_generator, deepseek=deepseek_generator, cloud_enabled=CLOUD_GENERATOR_ENABLED
        )
        query_expander = QueryExpander(
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11435"),
            model=os.getenv("QUERY_EXPANDER_MODEL", "qwen2.5:7b")
        )

        try:
            langfuse = Langfuse(
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
            )
            LANGFUSE_ENABLED = True
            logger.info("Langfuse v2 connected")
        except Exception as e:
            langfuse = None
            LANGFUSE_ENABLED = False
            logger.warning(f"Langfuse disabled: {e}")

        vector_store.create_collection(vector_size=embedder.get_vector_size())
        init_db()  # documents_registry, file_hashes, folders_registry all load from Postgres here —
                   # no Qdrant scroll needed at startup (see documents table in init_db())

        # Everything above is deliberately synchronous and can take long
        # enough for Postgres to restart or the guard connection to disappear
        # while models are loading.  Re-prove the SAME session immediately
        # before readiness; otherwise this process could start serving for a
        # full watchdog interval on a lock Postgres had already released.
        try:
            await check_single_instance_guard(watchdog_timeout)
        except TimeoutError as e:
            # wait_for() timed out while libpq was still executing in a
            # worker thread.  Trying to unwind normally and close that same
            # connection can itself block behind the stuck query, so use the
            # same unconditional fail-stop policy as the steady-state
            # watchdog.  The OS tears down both the thread and socket.
            _mark_single_instance_unhealthy()
            logger.critical(
                "single-instance guard recheck timed out after %.3fs during "
                "startup — exiting immediately before accepting traffic: %s",
                watchdog_timeout, e,
            )
            os._exit(1)

        _single_instance_healthy = True
        _single_instance_watchdog_task = asyncio.create_task(watch_single_instance_guard(
            interval_seconds=watchdog_interval,
            timeout_seconds=watchdog_timeout,
            on_failure=_mark_single_instance_unhealthy,
        ))
        # Fire-and-forget: retries anything a previous crash left
        # half-finished (a pending delete, an unsynced folder rename). Not
        # awaited — see _startup_reconciliation_sweep()'s docstring for why
        # this must not block readiness.
        _startup_reconciliation_task = asyncio.create_task(_startup_reconciliation_sweep())
    except BaseException:
        # The guard already succeeded by this point, so without this the
        # process would still exit (uvicorn treats a startup exception as
        # fatal) and the kernel would eventually tear down the connection
        # and release the lock anyway — but only on ITS timeline, not
        # immediately. An explicit release here means the next instance
        # doesn't have to wait that out. See release_single_instance_guard()'s
        # docstring in lock.py.
        _single_instance_healthy = False
        if _startup_reconciliation_task is not None:
            _startup_reconciliation_task.cancel()
            try:
                await _startup_reconciliation_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning("Startup reconciliation task ended with an error during startup cleanup: %s", e)
            _startup_reconciliation_task = None
        if _single_instance_watchdog_task is not None:
            _single_instance_watchdog_task.cancel()
            try:
                await _single_instance_watchdog_task
            except asyncio.CancelledError:
                pass
            _single_instance_watchdog_task = None
        release_single_instance_guard()
        raise

    logger.info("RAG Knowledge Base API started")


async def shutdown():
    global _single_instance_healthy, _single_instance_watchdog_task, _startup_reconciliation_task
    _single_instance_healthy = False
    # No task that can mutate Postgres/Qdrant/disk may survive release of
    # the process-wide advisory lock.  Otherwise the successor process can
    # acquire the guard and load state while this old sweep is still writing.
    if _startup_reconciliation_task is not None:
        _startup_reconciliation_task.cancel()
        try:
            await _startup_reconciliation_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("Startup reconciliation task ended with an error during shutdown: %s", e)
        _startup_reconciliation_task = None
    if _single_instance_watchdog_task is not None:
        _single_instance_watchdog_task.cancel()
        try:
            await _single_instance_watchdog_task
        except asyncio.CancelledError:
            pass
        _single_instance_watchdog_task = None
    release_single_instance_guard()



# /upload, /upload-batch, /query, /query/stream, /documents, /folders/*,
# /documents/{id}/pages/{page_num}, /models, and /health* — all split out
# to api/upload.py, api/query.py, api/documents.py, and api/health.py
# purely to shrink this file (see the refactor plan). /pdf/{doc_id} does
# its own manual auth (documents.public_router); /models is protected but
# /health* are deliberately public — hence two routers per module rather
# than one, see each module's own docstring for why.
protected.include_router(upload.router)
protected.include_router(query.router)
protected.include_router(documents.protected_router)
protected.include_router(health.protected_router)
app.include_router(protected)

app.include_router(documents.public_router)
app.include_router(health.public_router)
