"""
RAG Knowledge Base API
"""
import logging
import math
import uuid
import time
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv
load_dotenv()

from api.middleware import MaxBodySizeMiddleware
from api.schemas import ChatTurn, QueryRequest, QueryResponse
from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException, Security, Depends, Header
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
import json
import aiofiles
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor

from langfuse import Langfuse

from ingestion.pdf_parser import PDFParser
from ingestion.txt_parser import TxtParser, decode_text_file
from ingestion.chunker import SmartChunker, normalize_whitespace, chunk_context_text
from lock import (
    shared_lock,
    acquire_single_instance_guard,
    check_single_instance_guard,
    release_single_instance_guard,
    watch_single_instance_guard,
)
from embeddings.embedding_service import EmbeddingService
from vector_db.qdrant_client import VectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rag.retriever import HybridRetriever, promote_identity_matches, promote_document_opening_chunks, promote_missing_compare_documents
from rag.executors import run_on_gpu
from rag.reranker import CrossEncoderReranker, SimpleReranker
from rag.prompt_builder import PromptBuilder
from rag.generator import LLMGenerator, PartialStreamError
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

async def require_not_backing_up():
    """FastAPI dependency — holds lock.py's SHARED flock for the entire
    request, not just a check at the start of it. A plain existence-check
    of a flag file (an earlier approach) is a gate, not mutual exclusion:
    checked once, an upload that passed a moment before backup created its
    lock file kept mutating Qdrant/Postgres/disk for the rest of its own
    duration, invisible to backup_qdrant.sh's EXCLUSIVE flock hold. Holding
    a real kernel lock for the whole request means backup's exclusive
    acquire (a bounded `flock -w` wait — see backup_qdrant.sh) genuinely
    waits for every in-flight mutation to finish first instead of just
    giving up, and any NEW mutation that starts after backup already holds
    the exclusive lock fails fast with 503 instead of being let through.

    BlockingIOError is caught ONLY around acquiring the lock (the
    __enter__ call, before yield) — not wrapped around `yield` itself. A
    `with shared_lock(): yield` here would also catch a BlockingIOError
    the endpoint's own body happened to raise for an unrelated reason (a
    non-blocking read on some other fd, say) and misreport it as "backup
    in progress" instead of letting it surface as whatever error it
    actually is."""
    cm = shared_lock()
    try:
        cm.__enter__()
    except BlockingIOError:
        raise HTTPException(503, "Backup in progress — try again in a few minutes")
    try:
        yield
    finally:
        cm.__exit__(None, None, None)

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
# batch request could contain, no cap on PDF page count (so OCR time scaled
# with an attacker/mistake-controlled input), and no bound on how many
# uploads could be parsing/OCR-ing/embedding at once. MAX_CONCURRENT_
# INGESTIONS only needs to be small — embedding is already serialized onto
# one GPU worker by run_on_gpu (rag/executors.py), but PDF parsing/OCR is
# CPU-bound and runs via plain asyncio.to_thread, so nothing else limited
# how many of those could run in parallel before this.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_BATCH_FILES = int(os.getenv("MAX_BATCH_FILES", "20"))
MAX_CONCURRENT_INGESTIONS = int(os.getenv("MAX_CONCURRENT_INGESTIONS", "2"))
_ingestion_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INGESTIONS)
_active_ingestions = 0  # only ever mutated between an `await` boundary and the next, see _ingestion_slot() — safe without a separate lock under asyncio's single-threaded cooperative scheduling


@asynccontextmanager
async def _ingestion_slot(doc_id: str):
    """Wraps _ingestion_semaphore with observability: logs how many
    ingestion jobs are actually concurrently in this block right now, not
    just that a semaphore object exists — the thing worth being able to
    verify from logs under real concurrent load, not just trust from
    reading the semaphore size."""
    global _active_ingestions
    async with _ingestion_semaphore:
        _active_ingestions += 1
        logger.info(f"Ingestion slot acquired for {doc_id} ({_active_ingestions}/{MAX_CONCURRENT_INGESTIONS} in use)")
        try:
            yield
        finally:
            _active_ingestions -= 1
            logger.info(f"Ingestion slot released for {doc_id} ({_active_ingestions}/{MAX_CONCURRENT_INGESTIONS} in use)")

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

# Every supported format needs an entry here for the streaming upload
# helper (below) to reject an obvious mismatch (a renamed .exe as
# "report.pdf", say) before it ever reaches PyMuPDF/the parser — checked
# against just the first chunk read off the wire, not the whole file.
# TXT has no fixed signature, so any content passes for it; ".txt" is a
# labeling convention, not a format PyMuPDF or anything else parses
# structurally, so there is nothing meaningful to check.
_FORMAT_MAGIC_BYTES = {"pdf": b"%PDF-"}


def _validate_signature(doc_format: str, header: bytes) -> bool:
    magic = _FORMAT_MAGIC_BYTES.get(doc_format)
    return magic is None or header.startswith(magic)


async def _stream_upload_to_disk(file: UploadFile, dest: Path, doc_format: str, max_bytes: int) -> str:
    """Streams the upload into a per-attempt temp file in fixed-size chunks,
    hashing incrementally — never holds more than one chunk in memory
    regardless of the file's actual size, unlike the previous
    `content = await file.read()` (which bought a full copy into RAM before
    anything else even looked at it). Only `os.replace()`s the temp file
    onto `dest` (atomic on the same filesystem — same directory guarantees
    that) once the whole upload has validated clean; any failure path only
    ever unlinks the temp file, never `dest` itself. Two things this
    protects against that writing straight to `dest` wouldn't:
      - `dest` never exists in a partially-written state for anything else
        (a directory listing, a concurrent request) to observe.
      - cleanup on failure can never delete a file that isn't this
        attempt's own — the temp name is unique per call, so there is no
        path under which "roll back this failed upload" could touch
        something that existed before it started.

    Dedup (`file_hash in file_hashes`) has to happen AFTER this — the full
    hash needs the full file, and there's no way to know it without reading
    everything, which is exactly the read this function makes safe to do."""
    tmp_path = dest.parent / f".upload_{uuid.uuid4().hex}.tmp"
    hasher = hashlib.md5()
    total = 0
    chunk_size = 1024 * 1024
    first_chunk = True
    error: Optional[HTTPException] = None

    try:
        async with aiofiles.open(tmp_path, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                if first_chunk:
                    first_chunk = False
                    if not _validate_signature(doc_format, chunk):
                        error = HTTPException(400, f"File content doesn't look like a {doc_format.upper()} file")
                        break
                total += len(chunk)
                if total > max_bytes:
                    error = HTTPException(413, f"File exceeds the {max_bytes // (1024 * 1024)}MB upload limit")
                    break
                hasher.update(chunk)
                await f.write(chunk)

        if error is None and total == 0:
            error = HTTPException(422, "Uploaded file is empty")
        if error is not None:
            raise error

        os.replace(tmp_path, dest)  # atomic — dest either doesn't exist yet or is fully this upload's content, never a partial write
    except BaseException:
        # BaseException, not Exception — a client disconnect while this
        # request is awaiting `file.read()` surfaces as asyncio.CancelledError,
        # which does NOT subclass Exception (since Python 3.8). An `except
        # Exception:` here would silently leak the temp file on every
        # cancelled upload; the bare `raise` below still propagates
        # cancellation (and anything else) to the caller unchanged —
        # cleanup runs, nothing is swallowed.
        tmp_path.unlink(missing_ok=True)
        raise
    return hasher.hexdigest()


def pick_parser(filename: str):
    """Returns (parser_instance, format) for a supported extension, or (None, None)."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    return PARSERS_BY_EXT.get(ext, (None, None))


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
        generator = LLMGenerator(
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11435"),
            model=os.getenv("LLM_MODEL", "qwen2.5:7b"),
            temperature=float(os.getenv("TEMPERATURE", "0.1"))
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


def _get_active_document(doc_id: str) -> Optional[dict]:
    """Return a document only while it is logically present."""
    doc = documents_registry.get(doc_id)
    if doc is None or doc.get("status", "active") != "active":
        return None
    return doc


def _validate_query_document_scope(request: QueryRequest):
    """Reject explicit references to missing/tombstoned documents."""
    requested = []
    if request.document_id:
        requested.append(request.document_id)
    if request.document_ids:
        requested.extend(request.document_ids)
    missing = list(dict.fromkeys(doc_id for doc_id in requested if _get_active_document(doc_id) is None))
    if missing:
        raise HTTPException(404, f"Document(s) not found: {', '.join(missing)}")


def _active_chunks(chunks: list[dict]) -> list[dict]:
    """Drop stale Qdrant points belonging to logical tombstones."""
    return [chunk for chunk in chunks if _get_active_document(str(chunk.get("document_id", ""))) is not None]


async def _rollback_upload(doc_id: str, file_path: Path):
    """Best-effort compensating cleanup for an ingestion that failed
    somewhere along parse -> chunk -> embed -> upsert -> db_save. Called
    from every one of those failure paths, so it can't assume how far a
    given attempt actually got — Qdrant points, Postgres rows, and the file
    might each be present or absent in any combination. All three are
    therefore attempted unconditionally and independently (a DELETE/
    unlink for something that was never written is just a no-op), each in
    its own try/except logged separately.

    None of the three re-raises: this runs from inside the caller's
    `except` block for the REAL failure, and letting a cleanup error
    propagate from here would replace that original exception with a
    confusing one about cleanup instead — the caller's own logging/
    HTTPException already captured what actually went wrong before calling
    this."""
    try:
        await asyncio.to_thread(
            vector_store.client.delete,
            collection_name=vector_store.collection,
            points_selector=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]),
        )
    except Exception as e:
        logger.error(f"Rollback: Qdrant cleanup failed for {doc_id}: {e}")

    try:
        def _delete_doc_rows():
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM file_hashes WHERE doc_id = %s", (doc_id,))
                # document_pages cascades via its FK's ON DELETE CASCADE (init_db()).
                cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
        await asyncio.to_thread(_delete_doc_rows)
    except Exception as e:
        logger.error(f"Rollback: Postgres cleanup failed for {doc_id}: {e}")

    try:
        file_path.unlink(missing_ok=True)
    except Exception as e:
        logger.error(f"Rollback: file cleanup failed for {doc_id} ({file_path}): {e}")


@protected.post("/upload", dependencies=[Depends(require_not_backing_up)])
async def upload_document(file: UploadFile = File(...), folder: str = Form("")):
    safe_filename = Path(file.filename).name  # strip any path components
    doc_parser, doc_format = pick_parser(safe_filename)
    if doc_parser is None:
        raise HTTPException(400, "Only PDF and TXT files are supported")

    doc_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{doc_id}_{safe_filename}"

    # Streams straight to disk (see _stream_upload_to_disk) instead of
    # buffering the whole upload in RAM first — dedup still needs the full
    # hash, so it's checked here, after the file is already safely on disk.
    file_hash = await _stream_upload_to_disk(file, file_path, doc_format, MAX_UPLOAD_BYTES)

    if file_hash in file_hashes:
        file_path.unlink(missing_ok=True)
        existing_id = file_hashes[file_hash]
        existing = documents_registry.get(existing_id, {})
        raise HTTPException(409, f"File already uploaded as '{existing.get('filename', existing_id)}' (id: {existing_id})")

    # Everything from here through the Qdrant upsert is one unit: any
    # failure — parse, chunk, embed, or upsert — must leave neither an
    # orphaned file nor orphaned Qdrant points behind, so it's all one
    # try/except around _rollback_upload rather than the empty-chunks case
    # having its own bespoke cleanup and everything else having none.
    # The semaphore bounds concurrent CPU-bound parse/OCR + GPU-bound embed
    # work — embedding is already serialized onto one GPU worker by
    # run_on_gpu, but nothing previously capped how many uploads could be
    # parsing/OCR-ing in parallel via plain asyncio.to_thread.
    try:
        async with _ingestion_slot(doc_id):
            t_parse = time.time()
            parsed = await asyncio.to_thread(doc_parser.parse, str(file_path))
            parse_ms = int((time.time() - t_parse) * 1000)
            chunks = chunker.chunk_document(parsed.pages, doc_id)

            if not chunks:
                raise ValueError("Could not extract text from document")

            for c in chunks:
                c.filename = safe_filename
                c.pages = parsed.total_pages
                c.folder = folder or ""

            texts = [chunk_context_text(c) for c in chunks]
            t_embed = time.time()
            vectors = await run_on_gpu(embedder.embed_batch, texts)
            embed_ms = int((time.time() - t_embed) * 1000)
            await asyncio.to_thread(vector_store.upsert_chunks, chunks, vectors)
    except asyncio.CancelledError:
        # CancelledError does NOT subclass Exception (since Python 3.8) —
        # a client disconnect or server shutdown cancelling this task while
        # it's mid-parse/embed/upsert would otherwise skip both except
        # clauses below entirely and leave the file/Qdrant points orphaned.
        # Must re-raise bare (not as an HTTPException) — swallowing
        # cancellation here would make the task look like it completed
        # normally instead of actually being cancelled.
        logger.warning(f"Ingestion cancelled for {doc_id} ({safe_filename}) — rolling back")
        await _rollback_upload(doc_id, file_path)
        raise
    except ValueError as e:
        await _rollback_upload(doc_id, file_path)
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error(f"Ingestion failed for {doc_id} ({safe_filename}): {e}")
        await _rollback_upload(doc_id, file_path)
        raise HTTPException(500, f"Failed to process document: {e}")

    ocr_pages = sum(1 for p in parsed.pages if p.get("has_ocr"))
    logger.info(
        f"Ingestion: {safe_filename} | pages={parsed.total_pages} ocr={ocr_pages} "
        f"chunks={len(chunks)} parse_ms={parse_ms} embed_ms={embed_ms}"
    )
    if LANGFUSE_ENABLED:
        try:
            langfuse.trace(name="doc_ingestion", input=safe_filename, tags=["upload"],
                           metadata={"doc_id": doc_id, "pages": parsed.total_pages,
                                     "ocr_pages": ocr_pages, "chunks": len(chunks),
                                     "size_kb": parsed.file_size_kb, "folder": folder or "",
                                     "parse_ms": parse_ms, "embed_ms": embed_ms})
            langfuse.flush()
        except Exception:
            pass

    doc_meta = {
        "doc_id": doc_id,
        "filename": safe_filename,
        "pages": parsed.total_pages,
        "chunks": len(chunks),
        "size_kb": parsed.file_size_kb,
        "metadata": parsed.metadata,
        "folder": folder or "",
        "format": doc_format,
    }
    # file_hashes/folders_registry/documents_registry are only updated in
    # memory AFTER db_save_ingestion() commits — updating them earlier (the
    # old code set file_hashes[file_hash] before this could even fail) left
    # a hash entry with nothing behind it if the save failed: every retry
    # of the same file then hit the file_hash-seen check below and got a
    # false 409, even though the document/file/Qdrant points were all gone.
    try:
        await asyncio.to_thread(db_save_ingestion, doc_meta, parsed.pages, file_hash)
    except DuplicateFileHashError:
        logger.warning(f"Concurrent upload race for {doc_id} (file_hash already claimed) — discarding this attempt")
        await _rollback_upload(doc_id, file_path)
        existing_id = file_hashes.get(file_hash)
        existing = documents_registry.get(existing_id, {}) if existing_id else {}
        raise HTTPException(409, f"File already uploaded as '{existing.get('filename', existing_id or file_hash)}'")
    except Exception as e:
        logger.error(f"DB save failed for {doc_id}, rolling back Qdrant points + uploaded file: {e}")
        await _rollback_upload(doc_id, file_path)
        raise HTTPException(500, f"Failed to save document — upload did not complete: {e}")
    file_hashes[file_hash] = doc_id
    if folder:
        folders_registry.add(folder)
    documents_registry[doc_id] = doc_meta

    return {
        "doc_id": doc_id,
        "filename": safe_filename,
        "pages": parsed.total_pages,
        "chunks_created": len(chunks),
        "status": "indexed"
    }


@protected.post("/upload-batch", dependencies=[Depends(require_not_backing_up)])
async def upload_batch(files: list[UploadFile] = File(...), folder: str = Form("")):
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(413, f"Batch exceeds the {MAX_BATCH_FILES}-file limit ({len(files)} files sent)")

    results = []
    for file in files:
        doc_id = None   # reset each iteration — a stale doc_id/file_path from
        file_path = None  # a previous file must never be used for this file's cleanup
        try:
            safe_name = Path(file.filename).name
            doc_parser, doc_format = pick_parser(safe_name)
            if doc_parser is None:
                results.append({"filename": safe_name, "status": "error", "error": "Only PDF and TXT files are supported"})
                continue

            doc_id = str(uuid.uuid4())
            file_path = UPLOAD_DIR / f"{doc_id}_{safe_name}"

            # Streams straight to disk (see _stream_upload_to_disk) instead
            # of buffering the whole upload in RAM — a size/signature
            # failure here already cleaned up its own file, so it's reported
            # per-file (not a whole-batch abort) without any rollback call.
            try:
                file_hash = await _stream_upload_to_disk(file, file_path, doc_format, MAX_UPLOAD_BYTES)
            except HTTPException as e:
                results.append({"filename": safe_name, "status": "error", "error": e.detail})
                continue

            if file_hash in file_hashes:
                file_path.unlink(missing_ok=True)
                existing_id = file_hashes[file_hash]
                existing = documents_registry.get(existing_id, {})
                results.append({"filename": safe_name, "status": "skipped", "error": f"Already uploaded as '{existing.get('filename', existing_id)}'"})
                continue

            # Bounds concurrent CPU-bound parse/OCR + GPU-bound embed work
            # across all in-flight uploads (single + batch share the same
            # semaphore) — see MAX_CONCURRENT_INGESTIONS above.
            async with _ingestion_slot(doc_id):
                parsed = await asyncio.to_thread(doc_parser.parse, str(file_path))
                chunks = chunker.chunk_document(parsed.pages, doc_id)

                if not chunks:
                    raise ValueError("Could not extract text")

                for c in chunks:
                    c.filename = safe_name
                    c.pages = parsed.total_pages
                    c.folder = folder or ""

                texts = [chunk_context_text(c) for c in chunks]
                vectors = await run_on_gpu(embedder.embed_batch, texts)
                await asyncio.to_thread(vector_store.upsert_chunks, chunks, vectors)

            doc_meta = {
                "doc_id": doc_id,
                "filename": safe_name,
                "pages": parsed.total_pages,
                "chunks": len(chunks),
                "size_kb": parsed.file_size_kb,
                "metadata": parsed.metadata,
                "folder": folder or "",
                "format": doc_format,
            }
            # file_hashes/folders_registry/documents_registry only updated in
            # memory after a successful commit — see db_save_ingestion().
            await asyncio.to_thread(db_save_ingestion, doc_meta, parsed.pages, file_hash)
            file_hashes[file_hash] = doc_id
            if folder:
                folders_registry.add(folder)
            documents_registry[doc_id] = doc_meta
            results.append({"doc_id": doc_id, "filename": safe_name, "status": "indexed",
                            "pages": parsed.total_pages, "chunks_created": len(chunks)})

        except DuplicateFileHashError:
            logger.warning(f"Concurrent upload race for {doc_id} (file_hash already claimed) — discarding this attempt")
            if doc_id and file_path:
                await _rollback_upload(doc_id, file_path)
            existing_id = file_hashes.get(file_hash)
            existing = documents_registry.get(existing_id, {}) if existing_id else {}
            results.append({"filename": safe_name, "status": "skipped",
                            "error": f"Already uploaded as '{existing.get('filename', existing_id or file_hash)}'"})

        except asyncio.CancelledError:
            # Does NOT subclass Exception (Python 3.8+) — must stay its own
            # clause or a disconnect mid-file would fall through to the
            # generic handler below, get reported as a normal per-file
            # "error" result, and let the loop carry on to the next file
            # instead of actually stopping. Rolls back this file only, then
            # re-raises to abort the whole batch — the caller is gone, there
            # is no results list left to return it to.
            logger.warning(f"Batch ingestion cancelled for {file.filename} — rolling back and aborting batch")
            if doc_id and file_path:
                await _rollback_upload(doc_id, file_path)
            raise

        except Exception as e:
            logger.error(f"Batch error {file.filename}: {e}")
            if doc_id and file_path:
                await _rollback_upload(doc_id, file_path)
            results.append({"filename": file.filename, "status": "error", "error": str(e)})

    indexed = sum(1 for r in results if r["status"] == "indexed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    return {"total": len(results), "indexed": indexed, "skipped": skipped, "errors": errors, "results": results}


def _augment_compare_queries(expanded_queries: list[str], document_ids: Optional[list[str]]) -> list[str]:
    # query_expander.expand() decomposes within an 80-token LLM budget — for
    # a "compare A, B, C" question that spells out every filename plus
    # per-aspect instructions, it commonly runs out of budget after the
    # first document's name (observed: all sub-queries referenced only the
    # first of two compared documents). doc_filter in hybrid_search only
    # restricts which documents are *eligible*, it doesn't pull one in on
    # its own — a document never named in any query text simply never
    # enters the candidate pool, so retrieve_expanded's "at least 1 chunk
    # per document" guarantee has nothing to promote for it. One
    # deterministic query per named document closes that gap regardless of
    # what the free-form expander produced.
    if not document_ids or len(document_ids) < 2:
        return expanded_queries
    for doc_id in document_ids:
        filename = documents_registry.get(doc_id, {}).get("filename")
        if filename:
            expanded_queries.append(f"main subject, parties, and conclusions of {filename}")
    return expanded_queries


@protected.post("/query", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    query_expander=Depends(get_query_expander),
    retriever=Depends(get_retriever),
    reranker=Depends(get_reranker),
    prompt_builder=Depends(get_prompt_builder),
    generator=Depends(get_generator),
):
    # Blank/whitespace-only question is rejected by QueryRequest's own
    # validator (api/schemas.py) before this body ever runs — see
    # _question_not_blank.
    _validate_query_document_scope(request)

    if _query_semaphore.locked():
        raise HTTPException(429, "Too many concurrent requests, please try again shortly")

    async with _query_semaphore:
        return await _do_query(request, query_expander, retriever, reranker, prompt_builder, generator)


async def _do_query(request: QueryRequest, query_expander, retriever, reranker, prompt_builder, generator):
    start_time = time.time()

    trace = None
    if LANGFUSE_ENABLED:
        try:
            trace = langfuse.trace(name="rag_query", input=request.question, tags=["query"])
        except Exception as e:
            logger.warning(f"Langfuse trace failed: {e}")

    # chat_history is otherwise only used by prompt_builder.build() below,
    # for the final ANSWER generation — retrieval/reranking never saw it at
    # all until contextualize(), so a follow-up like "What was the court's
    # final decision?" used to search/rerank on that literal text alone,
    # with nothing pointing retrieval at which case "the court" even means.
    # search_query is what actually goes to expand()/retrieve_expanded()/
    # rerank() below; request.question (the user's literal wording) still
    # goes to prompt_builder and the trace, unchanged.
    chat_history_dicts = [t.model_dump() for t in request.chat_history] if request.chat_history else []
    search_query = await query_expander.contextualize(request.question, chat_history_dicts)

    t0 = time.time()
    expanded_queries = await query_expander.expand(search_query)
    expanded_queries = _augment_compare_queries(expanded_queries, request.document_ids)
    chunks = await retriever.retrieve_expanded(expanded_queries, top_k=max(20, request.top_k * 5), folder=request.folder or None, document_ids=request.document_ids or None)
    chunks = _active_chunks(chunks)
    retrieval_ms = int((time.time() - t0) * 1000)

    if trace:
        try:
            trace.span(name="retrieval", input=request.question,
                      output={"chunks_found": len(chunks)}, metadata={"duration_ms": retrieval_ms})
        except Exception as e:
            logger.warning(f"Langfuse span failed: {e}")

    if not chunks:
        return QueryResponse(answer="No relevant information found in the knowledge base.",
                           sources=[], model=generator.model, tokens_used=0)

    top_chunks = await run_on_gpu(reranker.rerank, search_query, chunks, top_k=request.top_k)
    top_chunks = promote_identity_matches(chunks, top_chunks, RELEVANCE_THRESHOLD)
    top_chunks = promote_document_opening_chunks(chunks, top_chunks)
    top_chunks = promote_missing_compare_documents(chunks, top_chunks, request.document_ids)

    best_score = max((c.get("rerank_score", 0) for c in top_chunks), default=0)
    # RELEVANCE_THRESHOLD exists to catch "this isn't in the knowledge base at
    # all" for open-ended questions — moot when document_ids is set, since
    # the compare flow's scope already came from the user explicitly picking
    # real documents out of the library, not from the cross-encoder's opinion
    # of them. Observed directly: a legitimate 3-document compare (all three
    # in scope, all three retrieved — 60 candidates) refused outright because
    # the generic "for each document provide: 1)... 2)... 3)..." compare
    # template scores low against some documents' content even when that
    # content is exactly what was asked to compare. Gating a guaranteed-
    # in-scope answer on a threshold tuned for natural single-topic questions
    # (see eval/golden_dataset.json calibration) rejects real compares for no
    # benefit — there's no "not in the KB" case left to catch here.
    if best_score < RELEVANCE_THRESHOLD and not request.document_ids:
        logger.info(f"Best rerank score {best_score:.3f} below threshold {RELEVANCE_THRESHOLD} — not answering")
        return QueryResponse(answer="I couldn't find relevant information in the knowledge base to answer this question.",
                           sources=[], model=generator.model, tokens_used=0,
                           debug={"best_rerank_score": round(best_score, 4), "threshold": RELEVANCE_THRESHOLD,
                                  "chunks_retrieved": len(chunks), "chunks_after_rerank": len(top_chunks),
                                  "search_query": search_query if search_query != request.question else None})

    messages = prompt_builder.build(query=request.question, chunks=top_chunks,
                                   chat_history=chat_history_dicts)

    t1 = time.time()
    result = await generator.generate_with_refusal_retry(messages, model=request.model or None)
    generation_ms = int((time.time() - t1) * 1000)

    if trace:
        try:
            trace.generation(name="llm_generation", model=result["model"],
                           input=messages, output=result["answer"],
                           usage={"input": result.get("prompt_tokens", 0),
                                  "output": result.get("completion_tokens", 0),
                                  "total": result["total_tokens"]},
                           metadata={"duration_ms": generation_ms})
        except Exception as e:
            logger.warning(f"Langfuse generation failed: {e}")

    seen_docs = {}
    for c in top_chunks:
        doc_id = c.get("document_id")
        score = c.get("rerank_score", c.get("score", 0))
        if doc_id not in seen_docs or score > seen_docs[doc_id]["relevance_score"]:
            raw = c["text"].strip().replace("\n", " ")
            excerpt = raw[:150].rsplit(" ", 1)[0] + "…" if len(raw) > 150 else raw
            seen_docs[doc_id] = {
                "page": c.get("page_num"),
                "document": doc_id,
                "excerpt": excerpt,
                "chunk_text": raw,
                "relevance_score": round(score, 3),
                "char_start": c.get("char_start"),
                "char_end": c.get("char_end"),
            }
    sources = sorted(seen_docs.values(), key=lambda x: x["relevance_score"], reverse=True)

    total_ms = int((time.time() - start_time) * 1000)

    if trace:
        try:
            trace.update(output=result["answer"],
                        metadata={"total_ms": total_ms, "retrieval_ms": retrieval_ms,
                                  "generation_ms": generation_ms, "tokens_used": result["total_tokens"],
                                  "sources_count": len(sources)})
            langfuse.flush()
        except Exception as e:
            logger.warning(f"Langfuse update failed: {e}")

    return QueryResponse(answer=result["answer"], sources=sources,
                        model=result["model"], tokens_used=result["total_tokens"],
                        debug={
                            "search_query": search_query if search_query != request.question else None,
                            "expanded_queries": expanded_queries,
                            "retrieval_ms": retrieval_ms,
                            "generation_ms": generation_ms,
                            "total_ms": total_ms,
                            "chunks_retrieved": len(chunks),
                            "chunks_after_rerank": len(top_chunks),
                            "top_chunks": [
                                {
                                    "chunk_id": c.get("chunk_id", ""),
                                    "document_id": c.get("document_id", ""),
                                    "page_num": c.get("page_num", 0),
                                    "score": round(c.get("rerank_score", c.get("score", 0)), 4),
                                    "source": c.get("source", ""),
                                    "text_preview": c.get("text", "")[:100],
                                }
                                for c in top_chunks
                            ]
                        })


@protected.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    query_expander=Depends(get_query_expander),
    retriever=Depends(get_retriever),
    reranker=Depends(get_reranker),
    prompt_builder=Depends(get_prompt_builder),
    generator=Depends(get_generator),
):
    # Blank/whitespace-only question is rejected by QueryRequest's own
    # validator (api/schemas.py) before this body ever runs — see
    # _question_not_blank.
    # query_expander/retriever/reranker/prompt_builder/generator above are
    # captured by event_stream()'s closure below — no need to thread them
    # through explicitly the way _do_query() needs them passed in, since
    # event_stream is defined inside this same function.
    _validate_query_document_scope(request)

    if _query_semaphore.locked():
        raise HTTPException(429, "Too many concurrent requests, please try again shortly")
    await _query_semaphore.acquire()

    async def event_stream():
        trace = None
        try:
            start_time = time.time()

            if LANGFUSE_ENABLED:
                try:
                    trace = langfuse.trace(name="rag_stream", input=request.question, tags=["stream"])
                except Exception:
                    pass

            # See _do_query()'s identical comment: chat_history was otherwise
            # only used for the final answer generation below — retrieval/
            # reranking never saw it, so a context-dependent follow-up
            # ("What was the court's final decision?") searched/reranked on
            # that literal text alone, with nothing pointing at which case
            # "the court" refers to.
            chat_history_dicts = [t.model_dump() for t in request.chat_history] if request.chat_history else []
            search_query = await query_expander.contextualize(request.question, chat_history_dicts)

            t0 = time.time()
            expanded_queries = await query_expander.expand(search_query)
            expanded_queries = _augment_compare_queries(expanded_queries, request.document_ids)
            expansion_ms = int((time.time() - t0) * 1000)

            t1 = time.time()
            chunks = await retriever.retrieve_expanded(expanded_queries, top_k=max(20, request.top_k * 5), folder=request.folder or None, document_ids=request.document_ids or None)
            chunks = _active_chunks(chunks)
            retrieval_ms = int((time.time() - t1) * 1000)

            retrieval_scores = [c.get("score", 0) for c in chunks] if chunks else []
            retrieval_best = max(retrieval_scores) if retrieval_scores else 0
            score_meta = {
                "best": round(retrieval_best, 3),
                "avg": round(sum(retrieval_scores) / len(retrieval_scores), 3) if retrieval_scores else 0,
                "chunks_found": len(chunks),
                "queries_expanded": len(expanded_queries),
            }

            if trace:
                try:
                    trace.span(name="query_expansion", input=request.question,
                               output={"queries": expanded_queries},
                               metadata={"duration_ms": expansion_ms, "search_query": search_query})
                    trace.span(name="retrieval", input=expanded_queries,
                               output=score_meta,
                               metadata={"duration_ms": retrieval_ms})
                except Exception:
                    pass

            if not chunks:
                yield f"data: {json.dumps({'type': 'token', 'content': 'No relevant information found in the knowledge base.'})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            t2 = time.time()
            top_chunks = await run_on_gpu(reranker.rerank, search_query, chunks, top_k=request.top_k)
            top_chunks = promote_identity_matches(chunks, top_chunks, RELEVANCE_THRESHOLD)
            top_chunks = promote_document_opening_chunks(chunks, top_chunks)
            top_chunks = promote_missing_compare_documents(chunks, top_chunks, request.document_ids)
            rerank_ms = int((time.time() - t2) * 1000)
            reranker_type = type(reranker).__name__

            rerank_scores = [c.get("rerank_score", c.get("score", 0)) for c in top_chunks]
            best_score = max(rerank_scores) if rerank_scores else 0
            score_meta = {
                "best": round(best_score, 3),
                "avg": round(sum(rerank_scores) / len(rerank_scores), 3) if rerank_scores else 0,
                "chunks_found": len(chunks),
                "queries_expanded": len(expanded_queries),
            }

            if best_score < RELEVANCE_THRESHOLD and not request.document_ids:
                logger.info(f"Best rerank score {best_score:.3f} below threshold {RELEVANCE_THRESHOLD} — not answering")
                msg = "I couldn't find relevant information in the knowledge base to answer this question."
                yield f"data: {json.dumps({'type': 'token', 'content': msg})}\n\n"
                yield f"data: {json.dumps({'type': 'sources', 'sources': [], 'debug': {**score_meta, 'threshold': RELEVANCE_THRESHOLD, 'search_query': search_query if search_query != request.question else None}})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            messages = prompt_builder.build(query=request.question, chunks=top_chunks,
                                            chat_history=chat_history_dicts)

            t2 = time.time()
            answer_tokens = []
            async for token in generator.generate_stream_with_refusal_retry(messages, model=request.model or None):
                answer_tokens.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            generation_ms = int((time.time() - t2) * 1000)

            if trace:
                try:
                    trace.generation(name="llm_stream", model=request.model or generator.model,
                                     input=messages, output="".join(answer_tokens),
                                     metadata={"duration_ms": generation_ms})
                    trace.update(metadata={
                        "total_ms": int((time.time() - start_time) * 1000),
                        "expansion_ms": expansion_ms,
                        "retrieval_ms": retrieval_ms,
                        "rerank_ms": rerank_ms,
                        "generation_ms": generation_ms,
                        "reranker": reranker_type,
                        **score_meta,
                    })
                    langfuse.flush()
                except Exception:
                    pass

            # Best chunk per document from all retrieved chunks (not just top_chunks),
            # so compare queries always show sources for every document found.
            # Best chunk per document from top_chunks (reranker already decided relevance)
            seen_docs = {}
            for c in top_chunks:
                doc_id = c.get("document_id")
                score = c.get("rerank_score", c.get("score", 0))
                if doc_id not in seen_docs or score > seen_docs[doc_id]["relevance_score"]:
                    raw = c["text"].strip().replace("\n", " ")
                    excerpt = raw[:150].rsplit(" ", 1)[0] + "…" if len(raw) > 150 else raw
                    seen_docs[doc_id] = {
                        "page": c.get("page_num"),
                        "document": doc_id,
                        "excerpt": excerpt,
                        "chunk_text": raw,
                        "relevance_score": round(score, 3),
                        "char_start": c.get("char_start"),
                        "char_end": c.get("char_end"),
                    }
            sources = sorted(seen_docs.values(), key=lambda x: x["relevance_score"], reverse=True)

            total_ms = int((time.time() - start_time) * 1000)
            debug_payload = {
                'search_query': search_query if search_query != request.question else None,
                'expanded_queries': expanded_queries,
                'total_ms': total_ms,
                'expansion_ms': expansion_ms,
                'retrieval_ms': retrieval_ms,
                'rerank_ms': rerank_ms,
                'generation_ms': generation_ms,
                'chunks_retrieved': len(chunks),
                'chunks_after_rerank': len(top_chunks),
                'best_score': float(score_meta['best']),
                'avg_score': float(score_meta['avg']),
                'reranker': reranker_type,
                'model': request.model or generator.model,
                'top_chunks': [
                    {
                        'chunk_id': str(c.get('chunk_id', '')),
                        'document_id': str(c.get('document_id', '')),
                        'page_num': c.get('page_num'),
                        'score': float(c.get('rerank_score', c.get('score', 0))),
                        'source': str(c.get('source', '')),
                        'text_preview': str(c.get('text', ''))[:100],
                    }
                    for c in top_chunks
                ],
            }
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'debug': debug_payload})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except PartialStreamError as e:
            logger.error(str(e))
            if trace:
                try: trace.update(metadata={"error": "partial_stream", "chunks_sent": e.chunks_yielded})
                except Exception: pass
            yield f"data: {json.dumps({'type': 'error', 'error_type': 'partial_stream', 'message': 'The response was cut off midway. Please try again.', 'partial': True, 'chunks_sent': e.chunks_yielded})}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            if trace:
                try: trace.update(metadata={"error": str(e)})
                except Exception: pass
            yield f"data: {json.dumps({'type': 'error', 'message': 'Could not get a response from the model. Please try again.'})}\n\n"
        finally:
            _query_semaphore.release()

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@protected.get("/documents")
async def list_documents():
    # status='deleting' documents are durably-tombstoned but not yet fully
    # torn down (see delete_document()) — hidden here so a client never sees
    # something it can't act on, even though they still exist in
    # documents_registry for _reconcile_document_deletion()/the startup
    # sweep to find and finish.
    visible = [d for d in documents_registry.values() if d.get("status", "active") == "active"]
    return {"total": len(visible), "documents": visible,
            "folders": sorted(folders_registry)}


@protected.patch("/documents/{doc_id}/folder", dependencies=[Depends(require_not_backing_up)])
async def update_document_folder(doc_id: str, body: dict):
    async with _metadata_mutation_lock:
        doc = _get_active_document(doc_id)
        if doc is None:
            raise HTTPException(404, f"Document {doc_id} not found")
        folder = body.get("folder", "")

        # Postgres first, durably, before touching memory. Folder creation is
        # part of this same transaction, so failure leaves both registries
        # and the folders table untouched.
        await asyncio.to_thread(db_update_document_folder, doc_id, folder)
        if folder:
            folders_registry.add(folder)
        doc["folder"] = folder
        doc["folder_synced"] = False  # db_update_document_folder() just set this in Postgres too

        # Qdrant's copy is a derived index, synced best-effort — a failure
        # leaves folder_synced=False for later reconciliation.
        await _reconcile_document_folder_sync(doc_id)
        return {"doc_id": doc_id, "folder": folder, "qdrant_synced": doc["folder_synced"]}


@protected.get("/folders")
async def list_folders():
    # Merge folders from registry and from documents
    doc_folders = {
        d["folder"] for d in documents_registry.values()
        if d.get("status", "active") == "active" and d.get("folder")
    }
    all_folders = sorted(folders_registry | doc_folders)
    return {"folders": all_folders}


@protected.post("/folders", dependencies=[Depends(require_not_backing_up)])
async def create_folder(body: dict):
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Folder name required")
    # DB first — db_save_folder() now raises instead of swallowing, so a
    # failure here 500s before folders_registry is touched at all.
    async with _metadata_mutation_lock:
        await asyncio.to_thread(db_save_folder, name)
        folders_registry.add(name)
        return {"name": name}


@protected.delete("/folders/{name}", dependencies=[Depends(require_not_backing_up)])
async def delete_folder(name: str):
    # Same ordering as create_folder(): DB first, memory only after it
    # durably succeeds.
    async with _metadata_mutation_lock:
        await asyncio.to_thread(db_delete_folder, name)
        folders_registry.discard(name)
        return {"deleted": name}


@protected.patch("/folders/{name}", dependencies=[Depends(require_not_backing_up)])
async def rename_folder(name: str, body: dict):
    new_name = (body.get("name") or "").strip()
    if not new_name:
        raise HTTPException(400, "New name required")

    # Durable, atomic (one Postgres transaction — see db_rename_folder())
    # rename FIRST: this IS the operation's source of truth. It also marks
    # every affected document folder_synced=FALSE in that same transaction,
    # so even a crash right after this leaves a durable record of exactly
    # which documents still need their Qdrant payload updated — nothing
    # here depends on the per-document loop below completing.
    async with _metadata_mutation_lock:
        await asyncio.to_thread(db_rename_folder, name, new_name)

        folders_registry.discard(name)
        folders_registry.add(new_name)
        affected_doc_ids = [
            doc_id for doc_id, doc in documents_registry.items()
            if doc.get("status", "active") == "active" and doc.get("folder") == name
        ]
        for doc_id in affected_doc_ids:
            documents_registry[doc_id]["folder"] = new_name
            documents_registry[doc_id]["folder_synced"] = False

        # Best-effort from here — each document is reconciled independently,
        # and the mutation lock remains held through the derived-index writes.
        qdrant_sync_pending = []
        for doc_id in affected_doc_ids:
            await _reconcile_document_folder_sync(doc_id)
            if not documents_registry[doc_id].get("folder_synced", True):
                qdrant_sync_pending.append(doc_id)

        if qdrant_sync_pending:
            logger.warning(
                f"Folder rename {name!r} -> {new_name!r}: Postgres committed for all "
                f"{len(affected_doc_ids)} documents, but Qdrant sync failed for "
                f"{len(qdrant_sync_pending)} of them — will retry on next access/restart."
            )

        return {"old": name, "new": new_name, "documents_updated": len(affected_doc_ids),
                "qdrant_sync_pending": qdrant_sync_pending}


@protected.delete("/documents/{doc_id}", dependencies=[Depends(require_not_backing_up)])
async def delete_document(doc_id: str):
    """Deletes a document across Qdrant + Postgres + disk. Not a single
    atomic operation (there is no cross-store transaction that could make it
    one) — instead, status='deleting' in Postgres (see db_mark_document_
    deleting()) is written FIRST and durably, before anything else is
    touched, so a crash between any of the steps below always leaves a
    trail: either this call never got past the tombstone write (nothing
    happened, retry from scratch is correct), or it did, in which case
    _reconcile_document_deletion() picks up from wherever it left off —
    Qdrant delete-by-filter and unlink(missing_ok=True) are both safe to
    repeat on content that's already gone. Calling this endpoint again on a
    doc_id already marked 'deleting' is exactly that retry, and is also what
    the startup reconciliation sweep does automatically for anything a
    previous crash left unfinished."""
    async with _metadata_mutation_lock:
        doc = documents_registry.get(doc_id)
        if doc is None:
            raise HTTPException(404, f"Document {doc_id} not found")

        if doc.get("status", "active") == "active":
            await asyncio.to_thread(db_mark_document_deleting, doc_id)
            doc["status"] = "deleting"
        elif doc.get("status") != "deleting":
            raise HTTPException(404, f"Document {doc_id} not found")

        return await _reconcile_document_deletion(doc_id)


@protected.get("/documents/{doc_id}/pages/{page_num}")
async def get_document_page(doc_id: str, page_num: int):
    """Canonical source-viewing endpoint for both TXT and PDF documents —
    same response shape for both {document_id, format, page, total_pages,
    text, has_ocr} so the frontend needs no format branching — but the two
    formats source that text differently:

    - TXT: decoded + normalize_whitespace()'d straight off the uploaded
      file on every request, exactly like the old (now-removed)
      /documents/{doc_id}/content. TxtParser's output is a pure function of
      the file's bytes with no external library version to drift, so there
      is nothing here that a persisted copy would protect against — storing
      it in document_pages too would only add a failure mode (a TXT
      document that was never backfilled 404s) for no correctness benefit.
    - PDF: read from document_pages, persisted once at ingestion time (see
      db_save_ingestion()) — PyMuPDF/Tesseract extraction is NOT
      a pure function of just the file (library/config versions matter),
      so this is re-derived at ingestion and never again, keeping it in
      lockstep with the char_start/char_end offsets /query sources hand out.

    Superseded /documents/{doc_id}/content (TXT-only) and /pdf/{doc_id}/
    highlights (PyMuPDF page.search_for() against the PDF's own text layer)
    — the latter had no fallback for OCR'd pages (no embedded text layer to
    search) and, even on clean digital PDFs, could land boxes on visually
    blank space wherever the page's internal reading order didn't match its
    visual layout (e.g. text positioned near an inserted image). Offset-
    based lookup has none of that: it never touches the PDF's rendering."""
    doc = _get_active_document(doc_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    doc_format = doc.get("format", "pdf")
    total_pages = 1 if doc_format == "txt" else doc.get("pages", 1)
    if page_num < 1 or page_num > total_pages:
        raise HTTPException(404, f"Page {page_num} out of range (1-{total_pages})")

    if doc_format == "txt":
        txt_file = next(UPLOAD_DIR.glob(f"{doc_id}_*"), None)
        if not txt_file:
            raise HTTPException(404, "File not found on disk")

        def _read_normalized():
            return normalize_whitespace(decode_text_file(txt_file.read_bytes()))

        text = await asyncio.to_thread(_read_normalized)
        has_ocr = False
    else:
        def _fetch():
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT normalized_text, has_ocr FROM document_pages WHERE document_id = %s AND page_num = %s",
                    (doc_id, page_num),
                )
                return cur.fetchone()

        row = await asyncio.to_thread(_fetch)
        if not row:
            raise HTTPException(404, "Page text not indexed for this document — re-upload to backfill")
        text, has_ocr = row
        has_ocr = bool(has_ocr)

    return {
        "document_id": doc_id,
        "format": doc_format,
        "page": page_num,
        "total_pages": total_pages,
        "text": text,
        "has_ocr": has_ocr,
    }


@app.get("/pdf/{doc_id}")
async def get_pdf(doc_id: str, key: Optional[str] = None, x_api_key: Optional[str] = Header(default=None)):
    """PDF.js fetches this via XHR/fetch (not a raw browser navigation), so it
    can send X-API-Key like every other endpoint — prefer that over the
    ?key= query param, which lands in browser history and server access
    logs. The query param is kept only as a fallback for any caller that
    can't set a header (e.g. a plain <a href> opened in a new tab)."""
    from fastapi.responses import FileResponse
    if API_KEY and key != API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    doc = _get_active_document(doc_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    for f in UPLOAD_DIR.glob(f"{doc_id}_*"):
        return FileResponse(path=str(f), media_type="application/pdf",
                          filename=doc["filename"])
    raise HTTPException(404, "PDF file not found on disk")



@protected.get("/models")
async def list_models():
    """List available models from Ollama"""
    import httpx
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11435")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{ollama_url}/api/tags")
            if r.status_code == 200:
                data = r.json()
                models = []
                for m in data.get("models", []):
                    name = m.get("name", "")
                    size_bytes = m.get("size", 0)
                    size_gb = round(size_bytes / 1e9, 1) if size_bytes else 0
                    models.append({
                        "name": name,
                        "size_gb": size_gb,
                        "active": name == generator.model,
                    })
                return {"models": models, "current": generator.model}
    except Exception as e:
        logger.warning(f"Ollama models fetch failed: {e}")
    return {"models": [{"name": generator.model, "size_gb": 0, "active": True}],
            "current": generator.model}


app.include_router(protected)


async def _check_postgres(timeout: float = 3.0) -> bool:
    # connect_timeout only bounds the TCP connect phase; it does nothing
    # once a connection is established, so a hung/overloaded server could
    # still leave SELECT 1 blocking indefinitely. `options=-c
    # statement_timeout=...` sets a server-side bound on the query itself,
    # so the DB (not just our client) enforces it. asyncio.wait_for around
    # asyncio.to_thread below cannot force-stop the underlying OS thread —
    # it only stops the *caller* from waiting — so these two real,
    # server/libpq-enforced timeouts are what actually end _ping(); the
    # wait_for is just a backstop with headroom over them, not the primary
    # bound.
    def _ping():
        conn = psycopg2.connect(
            POSTGRES_URL,
            connect_timeout=int(timeout),
            options=f"-c statement_timeout={int(timeout * 1000)}",
        )
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        finally:
            conn.close()
    try:
        await asyncio.wait_for(asyncio.to_thread(_ping), timeout=timeout * 2 + 1)
        return True
    except Exception as e:
        logger.warning(f"Postgres health check failed: {e}")
        return False


async def _check_qdrant(timeout: float = QDRANT_REQUEST_TIMEOUT_SECONDS) -> dict | None:
    # get_collection_info() catches its own exceptions and returns
    # {"collection": ..., "error": str(e)} instead of raising — so a failure
    # here is only visible by checking for the "error" key, not via except.
    # vector_store's own client was built with timeout=QDRANT_REQUEST_TIMEOUT_SECONDS
    # (api/main.py's VectorStore(...) call) — that's the real, request-level
    # bound on the blocking HTTP call inside get_collection_info(). The
    # wait_for here can't stop that call once it's running in its thread, so
    # it's given headroom over the client's own timeout rather than racing it.
    try:
        info = await asyncio.wait_for(asyncio.to_thread(vector_store.get_collection_info), timeout=timeout + 2)
        return None if "error" in info else info
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")
        return None


async def _check_ollama(timeout: float = 3.0) -> bool:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{generator.ollama_url}/api/tags")
            return r.status_code == 200
    except Exception as e:
        logger.warning(f"Ollama health check failed: {e}")
        return False


async def _readiness() -> JSONResponse:
    # See _single_instance_healthy's declaration above startup(): flipped by
    # the single-instance watchdog the instant it loses the Postgres
    # advisory lock's session, immediately before the process exits. Checked
    # first and unconditionally — none of the checks below mean anything if
    # this process can no longer prove it's the only one holding
    # documents_registry/file_hashes/folders_registry. This one failure mode
    # stays a plain HTTPException (not the flat JSONResponse body below) —
    # it's "this process isn't even a valid replica", a different shape of
    # failure than "a dependency is down", so {"detail": "..."} is fine here.
    if not _single_instance_healthy:
        raise HTTPException(503, "single-instance guard lost its Postgres session — this process is exiting")

    qdrant_info, postgres_ok, ollama_ok = await asyncio.gather(
        _check_qdrant(), _check_postgres(), _check_ollama()
    )
    qdrant_ok = qdrant_info is not None
    all_ok = qdrant_ok and postgres_ok and ollama_ok
    body = {
        "status": "healthy" if all_ok else "degraded",
        "components": {
            "qdrant": "ok" if qdrant_ok else "error",
            "postgres": "ok" if postgres_ok else "error",
            "ollama": "ok" if ollama_ok else "error",
            "embedding_model": embedder.model_name,
            "llm_model": generator.model,
            "langfuse": "ok" if LANGFUSE_ENABLED else "disabled"
        },
        "vector_store": qdrant_info or {}
    }
    # Returned as a plain JSONResponse, not raised as an HTTPException(detail=body)
    # — FastAPI wraps HTTPException.detail in {"detail": ...}, which would nest
    # this body one level deeper than what it actually documents/looks like.
    return JSONResponse(status_code=200 if all_ok else 503, content=body)


@app.get("/health/live")
async def health_live():
    """Liveness: is this process itself still able to serve traffic at all —
    no calls to Postgres/Qdrant/Ollama. A load balancer/orchestrator should
    restart the process on failure here; a downstream dependency being down
    is NOT a liveness failure (see /health/ready for that)."""
    if not _single_instance_healthy:
        raise HTTPException(503, "single-instance guard lost its Postgres session — this process is exiting")
    return {"status": "alive"}


@app.get("/health/ready")
async def health_ready():
    """Readiness: process alive AND Postgres/Qdrant/Ollama actually reachable.
    Returns HTTP 503 (not 200 with a 'degraded' field buried in the body) the
    moment any of them fails, so orchestrators/load balancers can act on the
    status code alone."""
    return await _readiness()


@app.get("/health")
async def health_check():
    """Back-compat alias for /health/ready — same real checks, same 503 on
    failure. Kept because README/start_rag.sh/frontend/scripts already poll
    this path; new integrations should prefer /health/live or /health/ready
    directly."""
    return await _readiness()
