"""
RAG Knowledge Base API
"""
import logging
import uuid
import time
import asyncio
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv
load_dotenv()

from api.telegram import router as telegram_router
from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException, Security, Depends, Header
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json
import aiofiles
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor

from langfuse import Langfuse

from ingestion.pdf_parser import PDFParser
from ingestion.txt_parser import TxtParser, decode_text_file
from ingestion.chunker import SmartChunker, normalize_whitespace, chunk_context_text
from lock import shared_lock
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
        return  # ключ не задан — auth отключена
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

app = FastAPI(title="RAG Knowledge Base API", version="1.0.0")
app.include_router(telegram_router)  # без auth — Telegram сам вызывает

# Роутер для всех защищённых эндпоинтов
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

parser = PDFParser(ocr_language=os.getenv("PDF_OCR_LANGUAGE", "eng"))
txt_parser = TxtParser()
PARSERS_BY_EXT = {"pdf": (parser, "pdf"), "txt": (txt_parser, "txt")}


def pick_parser(filename: str):
    """Returns (parser_instance, format) for a supported extension, or (None, None)."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    return PARSERS_BY_EXT.get(ext, (None, None))


chunker = SmartChunker(
    chunk_size=int(os.getenv("MAX_CHUNK_SIZE", "512")),
    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50"))
)
embedder = EmbeddingService(model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
vector_store = VectorStore(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    collection=os.getenv("QDRANT_COLLECTION", "knowledge_base"),
    api_key=os.getenv("QDRANT_API_KEY"),
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

documents_registry: dict = {}
file_hashes: dict = {}

from contextlib import contextmanager

def get_db():
    return psycopg2.connect(os.getenv("POSTGRES_URL", "postgresql://raguser:ragpass@localhost:5432/ragdb"))

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
    try:
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
            cur.execute("SELECT doc_id, filename, pages, chunks, size_kb, metadata, folder, format FROM documents")
            for doc_id, filename, pages, chunks, size_kb, metadata, folder, doc_format in cur.fetchall():
                documents_registry[doc_id] = {
                    "doc_id": doc_id,
                    "filename": filename,
                    "pages": pages,
                    "chunks": chunks,
                    "size_kb": size_kb,
                    "metadata": metadata or {},
                    "folder": folder or "",
                    "format": doc_format or "pdf",
                }
        logger.info(
            f"DB ready | loaded {len(file_hashes)} hashes, {len(folders_registry)} folders, "
            f"{len(documents_registry)} documents"
        )
    except Exception as e:
        logger.warning(f"DB init failed: {e}")


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


def db_update_document_folder(doc_id: str, folder: str):
    try:
        with db_conn() as conn:
            conn.cursor().execute("UPDATE documents SET folder = %s WHERE doc_id = %s", (folder, doc_id))
    except Exception as e:
        logger.warning(f"DB update document folder failed: {e}")

def db_save_folder(name: str):
    try:
        with db_conn() as conn:
            conn.cursor().execute("INSERT INTO folders (name) VALUES (%s) ON CONFLICT DO NOTHING", (name,))
    except Exception as e:
        logger.warning(f"DB save folder failed: {e}")

def db_delete_folder(name: str):
    try:
        with db_conn() as conn:
            conn.cursor().execute("DELETE FROM folders WHERE name = %s", (name,))
    except Exception as e:
        logger.warning(f"DB delete folder failed: {e}")

def db_rename_folder(old_name: str, new_name: str):
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO folders (name) VALUES (%s) ON CONFLICT DO NOTHING", (new_name,))
            cur.execute("DELETE FROM folders WHERE name = %s", (old_name,))
            cur.execute("UPDATE documents SET folder = %s WHERE folder = %s", (new_name, old_name))
    except Exception as e:
        logger.warning(f"DB rename folder failed: {e}")

@app.on_event("startup")
async def startup():
    vector_store.create_collection(vector_size=embedder.get_vector_size())
    init_db()  # documents_registry, file_hashes, folders_registry all load from Postgres here —
               # no Qdrant scroll needed at startup (see documents table in init_db())
    logger.info("RAG Knowledge Base API started")


class ChatTurn(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    # Hard cap regardless of caller — retrieve_expanded() sizes its candidate
    # pool off this (max(20, top_k*5)), so an unbounded top_k (e.g. a
    # frontend "compare all documents in a 150-doc folder" bug computing
    # top_k = doc_count*4) turns into a multi-thousand-candidate rerank pass.
    # MAX_CONTEXT_CHARS (prompt_builder.py) already truncates the prompt to
    # ~3000 tokens regardless, so nothing above this cap could even reach
    # the LLM — it would just be wasted retrieval/rerank work.
    # Deliberately not Optional[int] — Optional[int] = Field(..., ge=1, le=20)
    # only enforces ge/le when a value is actually given; an explicit
    # "top_k": null in the request body still passes validation as None,
    # and request.top_k * 5 downstream then throws a bare TypeError -> 500.
    # A plain int with a default rejects null outright (422) instead.
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: Optional[str] = None
    # Restricts retrieval to exactly these documents — used by the frontend's
    # "compare N specific documents" flow. Previously that flow only ever
    # *hinted* at scope via filenames spelled out in the question text plus
    # a folder filter, which left hybrid search (and the reranker) free to
    # pull in other documents from the same folder — see
    # rag/retriever.py::retrieve_expanded's document_ids param.
    document_ids: Optional[list[str]] = None
    chat_history: Optional[list[ChatTurn]] = []
    model: Optional[str] = None
    rerank: Optional[bool] = True
    folder: Optional[str] = None
    channel: Optional[str] = None   # "telegram" | None (web)

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    model: str
    tokens_used: int
    debug: Optional[dict] = None


async def _rollback_upload(doc_id: str, file_path: Path):
    """Best-effort cleanup for a document whose Qdrant points/file were
    already written before db_save_ingestion() failed (for any reason,
    including the DuplicateFileHashError concurrency race) — nothing about
    this doc_id made it into Postgres, so it must not survive in Qdrant or
    on disk either."""
    try:
        await asyncio.to_thread(
            vector_store.client.delete,
            collection_name=vector_store.collection,
            points_selector=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]),
        )
    except Exception as cleanup_err:
        logger.error(f"Qdrant rollback also failed for {doc_id}: {cleanup_err}")
    file_path.unlink(missing_ok=True)


@protected.post("/upload", dependencies=[Depends(require_not_backing_up)])
async def upload_document(file: UploadFile = File(...), folder: str = Form("")):
    safe_filename = Path(file.filename).name  # strip any path components
    doc_parser, doc_format = pick_parser(safe_filename)
    if doc_parser is None:
        raise HTTPException(400, "Only PDF and TXT files are supported")

    content = await file.read()
    file_hash = hashlib.md5(content).hexdigest()

    if file_hash in file_hashes:
        existing_id = file_hashes[file_hash]
        existing = documents_registry.get(existing_id, {})
        raise HTTPException(409, f"File already uploaded as '{existing.get('filename', existing_id)}' (id: {existing_id})")

    doc_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{doc_id}_{safe_filename}"

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    t_parse = time.time()
    parsed = await asyncio.to_thread(doc_parser.parse, str(file_path))
    parse_ms = int((time.time() - t_parse) * 1000)
    chunks = chunker.chunk_document(parsed.pages, doc_id)

    if not chunks:
        file_path.unlink(missing_ok=True)
        raise HTTPException(422, "Could not extract text from document")

    for c in chunks:
        c.filename = safe_filename
        c.pages = parsed.total_pages
        c.folder = folder or ""

    texts = [chunk_context_text(c) for c in chunks]
    t_embed = time.time()
    vectors = await run_on_gpu(embedder.embed_batch, texts)
    embed_ms = int((time.time() - t_embed) * 1000)
    await asyncio.to_thread(vector_store.upsert_chunks, chunks, vectors)

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

            content = await file.read()
            file_hash = hashlib.md5(content).hexdigest()

            if file_hash in file_hashes:
                existing_id = file_hashes[file_hash]
                existing = documents_registry.get(existing_id, {})
                results.append({"filename": safe_name, "status": "skipped", "error": f"Already uploaded as '{existing.get('filename', existing_id)}'"})
                continue

            doc_id = str(uuid.uuid4())
            file_path = UPLOAD_DIR / f"{doc_id}_{safe_name}"

            async with aiofiles.open(file_path, "wb") as f_out:
                await f_out.write(content)

            parsed = await asyncio.to_thread(doc_parser.parse, str(file_path))
            chunks = chunker.chunk_document(parsed.pages, doc_id)

            if not chunks:
                file_path.unlink(missing_ok=True)
                results.append({"filename": safe_name, "status": "error", "error": "Could not extract text"})
                continue

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
async def query_knowledge_base(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    if _query_semaphore.locked():
        raise HTTPException(429, "Too many concurrent requests, please try again shortly")

    async with _query_semaphore:
        return await _do_query(request)


async def _do_query(request: QueryRequest):
    start_time = time.time()

    trace = None
    if LANGFUSE_ENABLED:
        try:
            trace = langfuse.trace(name="rag_query", input=request.question, tags=["query"])
        except Exception as e:
            logger.warning(f"Langfuse trace failed: {e}")

    t0 = time.time()
    expanded_queries = await query_expander.expand(request.question)
    expanded_queries = _augment_compare_queries(expanded_queries, request.document_ids)
    chunks = await retriever.retrieve_expanded(expanded_queries, top_k=max(20, request.top_k * 5), folder=request.folder or None, document_ids=request.document_ids or None)
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

    top_chunks = await run_on_gpu(reranker.rerank, request.question, chunks, top_k=request.top_k)
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
                                  "chunks_retrieved": len(chunks), "chunks_after_rerank": len(top_chunks)})

    messages = prompt_builder.build(query=request.question, chunks=top_chunks,
                                   chat_history=[t.model_dump() for t in request.chat_history] if request.chat_history else [],
                                   channel=request.channel)

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
async def query_stream(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")

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

            t0 = time.time()
            expanded_queries = await query_expander.expand(request.question)
            expanded_queries = _augment_compare_queries(expanded_queries, request.document_ids)
            expansion_ms = int((time.time() - t0) * 1000)

            t1 = time.time()
            chunks = await retriever.retrieve_expanded(expanded_queries, top_k=max(20, request.top_k * 5), folder=request.folder or None, document_ids=request.document_ids or None)
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
                               metadata={"duration_ms": expansion_ms})
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
            top_chunks = await run_on_gpu(reranker.rerank, request.question, chunks, top_k=request.top_k)
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
                yield f"data: {json.dumps({'type': 'sources', 'sources': [], 'debug': {**score_meta, 'threshold': RELEVANCE_THRESHOLD}})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            messages = prompt_builder.build(query=request.question, chunks=top_chunks,
                                            chat_history=[t.model_dump() for t in request.chat_history] if request.chat_history else [],
                                            channel=request.channel)

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
            yield f"data: {json.dumps({'type': 'error', 'error_type': 'partial_stream', 'message': 'Ответ оборвался на середине. Попробуйте повторить запрос.', 'partial': True, 'chunks_sent': e.chunks_yielded})}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            if trace:
                try: trace.update(metadata={"error": str(e)})
                except Exception: pass
            yield f"data: {json.dumps({'type': 'error', 'message': 'Не удалось получить ответ от модели. Попробуйте повторить запрос.'})}\n\n"
        finally:
            _query_semaphore.release()

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@protected.get("/documents")
async def list_documents():
    return {"total": len(documents_registry), "documents": list(documents_registry.values()),
            "folders": sorted(folders_registry)}


@protected.patch("/documents/{doc_id}/folder", dependencies=[Depends(require_not_backing_up)])
async def update_document_folder(doc_id: str, body: dict):
    if doc_id not in documents_registry:
        raise HTTPException(404, f"Document {doc_id} not found")
    folder = body.get("folder", "")
    documents_registry[doc_id]["folder"] = folder
    if folder:
        folders_registry.add(folder)
        await asyncio.to_thread(db_save_folder, folder)
    await asyncio.to_thread(db_update_document_folder, doc_id, folder)
    await asyncio.to_thread(
        vector_store.client.set_payload,
        collection_name=vector_store.collection,
        payload={"folder": folder},
        points=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))])
    )
    return {"doc_id": doc_id, "folder": folder}


@protected.get("/folders")
async def list_folders():
    # Merge folders from registry and from documents
    doc_folders = {d["folder"] for d in documents_registry.values() if d.get("folder")}
    all_folders = sorted(folders_registry | doc_folders)
    return {"folders": all_folders}


@protected.post("/folders", dependencies=[Depends(require_not_backing_up)])
async def create_folder(body: dict):
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Folder name required")
    folders_registry.add(name)
    db_save_folder(name)
    return {"name": name}


@protected.delete("/folders/{name}", dependencies=[Depends(require_not_backing_up)])
async def delete_folder(name: str):
    folders_registry.discard(name)
    db_delete_folder(name)
    return {"deleted": name}


@protected.patch("/folders/{name}", dependencies=[Depends(require_not_backing_up)])
async def rename_folder(name: str, body: dict):
    new_name = (body.get("name") or "").strip()
    if not new_name:
        raise HTTPException(400, "New name required")
    # Update all documents in this folder
    for doc_id, doc in documents_registry.items():
        if doc.get("folder") == name:
            doc["folder"] = new_name
            await asyncio.to_thread(
                vector_store.client.set_payload,
                collection_name=vector_store.collection,
                payload={"folder": new_name},
                points=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))])
            )
    folders_registry.discard(name)
    folders_registry.add(new_name)
    await asyncio.to_thread(db_rename_folder, name, new_name)  # also updates documents.folder in bulk
    return {"old": name, "new": new_name}


@protected.delete("/documents/{doc_id}", dependencies=[Depends(require_not_backing_up)])
async def delete_document(doc_id: str):
    if doc_id not in documents_registry:
        raise HTTPException(404, f"Document {doc_id} not found")

    filename = documents_registry[doc_id].get("filename", doc_id)
    errors = {}

    try:
        await asyncio.to_thread(
            vector_store.client.delete,
            collection_name=vector_store.collection,
            points_selector=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))])
        )
    except Exception as e:
        logger.error(f"Qdrant delete failed for {doc_id}: {e}")
        errors["qdrant"] = str(e)

    try:
        def _delete_doc_rows():
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM file_hashes WHERE doc_id = %s", (doc_id,))
                cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
        await asyncio.to_thread(_delete_doc_rows)
        for h in [h for h, d in file_hashes.items() if d == doc_id]:
            del file_hashes[h]
    except Exception as e:
        logger.error(f"DB delete failed for {doc_id}: {e}")
        errors["db"] = str(e)

    try:
        for f in UPLOAD_DIR.glob(f"{doc_id}_*"):
            f.unlink()
    except Exception as e:
        logger.error(f"File delete failed for {doc_id}: {e}")
        errors["file"] = str(e)

    if errors:
        # Keep in registry so the document remains visible and deletion can be retried
        logger.error(f"Document {doc_id} partially deleted, failed steps: {list(errors.keys())}")
        raise HTTPException(500, f"Partial deletion failure for {doc_id}. Failed steps: {list(errors.keys())}. Retry deletion.")

    del documents_registry[doc_id]
    return {"status": "deleted", "doc_id": doc_id, "filename": filename}


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
    doc = documents_registry.get(doc_id)
    if not doc:
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
    if doc_id not in documents_registry:
        raise HTTPException(404, "Document not found")
    for f in UPLOAD_DIR.glob(f"{doc_id}_*"):
        return FileResponse(path=str(f), media_type="application/pdf",
                          filename=documents_registry[doc_id]["filename"])
    raise HTTPException(404, "PDF file not found on disk")



@protected.get("/models")
async def list_models():
    """Список доступных моделей из Ollama"""
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


@app.get("/health")
async def health_check():
    qdrant_ok = False
    try:
        info = vector_store.get_collection_info()
        qdrant_ok = True
    except:
        info = {}
    return {
        "status": "healthy" if qdrant_ok else "degraded",
        "components": {
            "qdrant": "ok" if qdrant_ok else "error",
            "ollama": "ok",
            "embedding_model": embedder.model_name,
            "llm_model": generator.model,
            "langfuse": "ok" if LANGFUSE_ENABLED else "disabled"
        },
        "vector_store": info
    }
