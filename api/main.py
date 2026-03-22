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
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import aiofiles
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor

from langfuse import Langfuse

from ingestion.pdf_parser import PDFParser
from ingestion.chunker import SmartChunker
from embeddings.embedding_service import EmbeddingService
from vector_db.qdrant_client import VectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rag.retriever import HybridRetriever
from rag.reranker import CrossEncoderReranker, SimpleReranker
from rag.prompt_builder import PromptBuilder
from rag.generator import LLMGenerator
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

app = FastAPI(
    title="RAG Knowledge Base API",
    version="1.0.0",
    dependencies=[Depends(require_api_key)]
)
app.include_router(telegram_router, dependencies=[])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Minimum retrieval score to attempt an answer (0–1 cosine similarity scale).
# Below this threshold the knowledge base is considered to have no relevant content.
RELEVANCE_THRESHOLD = 0.30

# Max concurrent LLM requests — Ollama processes one at a time anyway,
# queuing more than this returns 429 immediately instead of timing out.
MAX_CONCURRENT_QUERIES = 3
_query_semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)

parser = PDFParser()
chunker = SmartChunker(chunk_size=512, chunk_overlap=50)
embedder = EmbeddingService()
vector_store = VectorStore()
retriever = HybridRetriever(embedder, vector_store)
try:
    reranker = CrossEncoderReranker()
except Exception as e:
    logger.warning(f"CrossEncoderReranker failed to load ({e}), falling back to SimpleReranker")
    reranker = SimpleReranker()
prompt_builder = PromptBuilder()
generator = LLMGenerator(ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11435"))
query_expander = QueryExpander(ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11435"))

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
                    doc_id VARCHAR(8) NOT NULL,
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
            cur.execute("SELECT hash, doc_id FROM file_hashes")
            for row in cur.fetchall():
                file_hashes[row[0]] = row[1]
            cur.execute("SELECT name FROM folders")
            for row in cur.fetchall():
                folders_registry.add(row[0])
        logger.info(f"DB ready | loaded {len(file_hashes)} hashes, {len(folders_registry)} folders")
    except Exception as e:
        logger.warning(f"DB init failed: {e}")

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
    except Exception as e:
        logger.warning(f"DB rename folder failed: {e}")

async def restore_from_qdrant():
    try:
        all_points = []
        offset = None
        while True:
            result = vector_store.client.scroll(
                collection_name=vector_store.collection,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            batch, next_offset = result
            all_points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset

        if not all_points:
            logger.info("Qdrant empty — nothing to restore")
            return

        doc_meta = {}
        bm25_chunks = []

        for point in all_points:
            p = point.payload or {}
            doc_id = p.get("document_id") or p.get("doc_id", "")
            filename = p.get("filename", "unknown")
            text = p.get("text", "")
            chunk_id = p.get("chunk_id", str(point.id))
            page_num = p.get("page_num", 0)
            chunk_index = p.get("chunk_index", 0)

            if text:
                bm25_chunks.append({
                    "text": text,
                    "chunk_id": chunk_id,
                    "page_num": page_num,
                    "document_id": doc_id,
                    "chunk_index": chunk_index,
                    "folder": p.get("folder", ""),
                })

            if doc_id:
                if doc_id not in doc_meta:
                    doc_meta[doc_id] = {
                        "doc_id": doc_id,
                        "filename": filename,
                        "pages": p.get("pages", 0),
                        "chunks": 0,
                        "size_kb": p.get("size_kb", 0),
                        "metadata": p.get("metadata", {}),
                        "folder": p.get("folder", ""),
                    }
                doc_meta[doc_id]["chunks"] += 1

        for doc_id, meta in doc_meta.items():
            documents_registry[doc_id] = meta

        if bm25_chunks:
            retriever.index_chunks_for_bm25(bm25_chunks)

        logger.info(f"Restored: {len(documents_registry)} docs, {len(bm25_chunks)} chunks in BM25")

    except Exception as e:
        logger.warning(f"Restore failed (non-critical): {e}")

@app.on_event("startup")
async def startup():
    vector_store.create_collection(vector_size=embedder.get_vector_size())
    init_db()
    await restore_from_qdrant()
    logger.info("RAG Knowledge Base API started")


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    document_id: Optional[str] = None
    chat_history: Optional[list] = []
    model: Optional[str] = None
    rerank: Optional[bool] = True
    folder: Optional[str] = None
    language: Optional[str] = None  # "en" | "ru" | None (auto)
    channel: Optional[str] = None   # "telegram" | None (web)

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    model: str
    tokens_used: int
    debug: Optional[dict] = None


@app.post("/upload")
async def upload_document(file: UploadFile = File(...), folder: str = Form("")):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    content = await file.read()
    file_hash = hashlib.md5(content).hexdigest()

    if file_hash in file_hashes:
        existing_id = file_hashes[file_hash]
        existing = documents_registry.get(existing_id, {})
        raise HTTPException(409, f"File already uploaded as '{existing.get('filename', existing_id)}' (id: {existing_id})")

    doc_id = str(uuid.uuid4())[:8]
    file_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    parsed = parser.parse(str(file_path))
    chunks = chunker.chunk_document(parsed.pages, doc_id)

    if not chunks:
        raise HTTPException(422, "Could not extract text from document")

    for c in chunks:
        c.filename = file.filename
        c.pages = parsed.total_pages
        c.folder = folder or ""

    texts = [c.text for c in chunks]
    vectors = embedder.embed_batch(texts)
    vector_store.upsert_chunks(chunks, vectors)

    bm25_chunks = [{"text": c.text, "chunk_id": c.chunk_id,
                    "page_num": c.page_num, "document_id": c.document_id,
                    "chunk_index": c.chunk_index, "filename": c.filename,
                    "folder": folder or ""}
                   for c in chunks]
    retriever.index_chunks_for_bm25(bm25_chunks)

    file_hashes[file_hash] = doc_id
    try:
        with db_conn() as conn:
            conn.cursor().execute(
                "INSERT INTO file_hashes (hash, doc_id, filename) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                (file_hash, doc_id, file.filename)
            )
    except Exception as e:
        logger.warning(f"DB save hash failed: {e}")

    if folder:
        folders_registry.add(folder)
        db_save_folder(folder)
    documents_registry[doc_id] = {
        "doc_id": doc_id,
        "filename": file.filename,
        "pages": parsed.total_pages,
        "chunks": len(chunks),
        "size_kb": parsed.file_size_kb,
        "metadata": parsed.metadata,
        "folder": folder or ""
    }

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "pages": parsed.total_pages,
        "chunks_created": len(chunks),
        "status": "indexed"
    }


@app.post("/upload-batch")
async def upload_batch(files: list[UploadFile] = File(...), folder: str = Form("")):
    results = []
    for file in files:
        try:
            if not file.filename.endswith(".pdf"):
                results.append({"filename": file.filename, "status": "error", "error": "Only PDF files supported"})
                continue

            content = await file.read()
            file_hash = hashlib.md5(content).hexdigest()

            if file_hash in file_hashes:
                existing_id = file_hashes[file_hash]
                existing = documents_registry.get(existing_id, {})
                results.append({"filename": file.filename, "status": "skipped", "error": f"Already uploaded as '{existing.get('filename', existing_id)}'"})
                continue

            doc_id = str(uuid.uuid4())[:8]
            file_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"

            async with aiofiles.open(file_path, "wb") as f_out:
                await f_out.write(content)

            parsed = parser.parse(str(file_path))
            chunks = chunker.chunk_document(parsed.pages, doc_id)

            if not chunks:
                results.append({"filename": file.filename, "status": "error", "error": "Could not extract text"})
                continue

            for c in chunks:
                c.filename = file.filename
                c.pages = parsed.total_pages
                c.folder = folder or ""

            texts = [c.text for c in chunks]
            vectors = embedder.embed_batch(texts)
            vector_store.upsert_chunks(chunks, vectors)

            new_bm25 = [{"text": c.text, "chunk_id": c.chunk_id,
                         "page_num": c.page_num, "document_id": c.document_id,
                         "chunk_index": c.chunk_index, "filename": c.filename,
                         "folder": folder or ""}
                        for c in chunks]
            retriever.index_chunks_for_bm25(retriever._bm25_chunks + new_bm25)

            file_hashes[file_hash] = doc_id
            try:
                with db_conn() as conn:
                    conn.cursor().execute(
                        "INSERT INTO file_hashes (hash, doc_id, filename) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                        (file_hash, doc_id, file.filename)
                    )
            except Exception as e:
                logger.warning(f"DB save hash failed: {e}")

            documents_registry[doc_id] = {
                "doc_id": doc_id,
                "filename": file.filename,
                "pages": parsed.total_pages,
                "chunks": len(chunks),
                "size_kb": parsed.file_size_kb,
                "metadata": parsed.metadata,
                "folder": folder or ""
            }
            results.append({"doc_id": doc_id, "filename": file.filename, "status": "indexed",
                            "pages": parsed.total_pages, "chunks_created": len(chunks)})

        except Exception as e:
            logger.error(f"Batch error {file.filename}: {e}")
            results.append({"filename": file.filename, "status": "error", "error": str(e)})

    indexed = sum(1 for r in results if r["status"] == "indexed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    return {"total": len(results), "indexed": indexed, "skipped": skipped, "errors": errors, "results": results}


@app.post("/query", response_model=QueryResponse)
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
    chunks = await retriever.retrieve_expanded(expanded_queries, top_k=max(20, request.top_k * 5), folder=request.folder or None)
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

    best_score = max(c.get("score", 0) for c in chunks)
    if best_score < RELEVANCE_THRESHOLD:
        logger.info(f"Best score {best_score:.3f} below threshold {RELEVANCE_THRESHOLD} — not answering")
        return QueryResponse(answer="I couldn't find relevant information in the knowledge base to answer this question.",
                           sources=[], model=generator.model, tokens_used=0)

    top_chunks = reranker.rerank(request.question, chunks, top_k=min(request.top_k, 3))

    messages = prompt_builder.build(query=request.question, chunks=top_chunks,
                                   chat_history=request.chat_history,
                                   language=request.language or None,
                                   channel=request.channel)

    t1 = time.time()
    result = await generator.generate(messages, model=request.model or None)
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

    seen_sources = set()
    sources = []
    for c in top_chunks:
        raw = c["text"].strip().replace("\n", " ")
        text_key = raw[:80].lower()
        if text_key in seen_sources:
            continue
        seen_sources.add(text_key)
        excerpt = raw[:150].rsplit(" ", 1)[0] + "…" if len(raw) > 150 else raw
        sources.append({
            "page": c.get("page_num"),
            "document": c.get("document_id"),
            "excerpt": excerpt,
            "chunk_text": raw,
            "relevance_score": round(c.get("rerank_score", c.get("score", 0)), 3)
        })

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


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    if _query_semaphore.locked():
        raise HTTPException(429, "Too many concurrent requests, please try again shortly")
    await _query_semaphore.acquire()

    async def event_stream():
        try:
            start_time = time.time()

            expanded_queries = await query_expander.expand(request.question)
            chunks = await retriever.retrieve_expanded(expanded_queries, top_k=max(20, request.top_k * 5), folder=request.folder or None)

            if not chunks:
                yield f"data: {json.dumps({'type': 'token', 'content': 'No relevant information found in the knowledge base.'})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            best_score = max(c.get("score", 0) for c in chunks)
            if best_score < RELEVANCE_THRESHOLD:
                logger.info(f"Best score {best_score:.3f} below threshold {RELEVANCE_THRESHOLD} — not answering")
                msg = "I couldn't find relevant information in the knowledge base to answer this question."
                yield f"data: {json.dumps({'type': 'token', 'content': msg})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            top_chunks = reranker.rerank(request.question, chunks, top_k=min(request.top_k, 3))
            messages = prompt_builder.build(query=request.question, chunks=top_chunks,
                                            chat_history=request.chat_history,
                                            language=request.language or None,
                                            channel=request.channel)

            async for token in generator.generate_stream(messages, model=request.model or None):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            seen_sources = set()
            sources = []
            for c in top_chunks:
                raw = c["text"].strip().replace("\n", " ")
                text_key = raw[:80].lower()
                if text_key in seen_sources:
                    continue
                seen_sources.add(text_key)
                excerpt = raw[:150].rsplit(" ", 1)[0] + "…" if len(raw) > 150 else raw
                sources.append({
                    "page": c.get("page_num"),
                    "document": c.get("document_id"),
                    "excerpt": excerpt,
                    "chunk_text": raw,
                    "relevance_score": round(c.get("rerank_score", c.get("score", 0)), 3)
                })

            total_ms = int((time.time() - start_time) * 1000)
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'debug': {'expanded_queries': expanded_queries, 'total_ms': total_ms, 'chunks_retrieved': len(chunks), 'chunks_after_rerank': len(top_chunks)}})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        finally:
            _query_semaphore.release()

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/documents")
async def list_documents():
    return {"total": len(documents_registry), "documents": list(documents_registry.values()),
            "folders": sorted(folders_registry)}


@app.patch("/documents/{doc_id}/folder")
async def update_document_folder(doc_id: str, body: dict):
    if doc_id not in documents_registry:
        raise HTTPException(404, f"Document {doc_id} not found")
    folder = body.get("folder", "")
    documents_registry[doc_id]["folder"] = folder
    if folder:
        folders_registry.add(folder)
        db_save_folder(folder)
    vector_store.client.set_payload(
        collection_name=vector_store.collection,
        payload={"folder": folder},
        points=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))])
    )
    return {"doc_id": doc_id, "folder": folder}


@app.get("/folders")
async def list_folders():
    # Merge folders from registry and from documents
    doc_folders = {d["folder"] for d in documents_registry.values() if d.get("folder")}
    all_folders = sorted(folders_registry | doc_folders)
    return {"folders": all_folders}


@app.post("/folders")
async def create_folder(body: dict):
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Folder name required")
    folders_registry.add(name)
    db_save_folder(name)
    return {"name": name}


@app.delete("/folders/{name}")
async def delete_folder(name: str):
    folders_registry.discard(name)
    db_delete_folder(name)
    return {"deleted": name}


@app.patch("/folders/{name}")
async def rename_folder(name: str, body: dict):
    new_name = (body.get("name") or "").strip()
    if not new_name:
        raise HTTPException(400, "New name required")
    # Update all documents in this folder
    for doc_id, doc in documents_registry.items():
        if doc.get("folder") == name:
            doc["folder"] = new_name
            vector_store.client.set_payload(
                collection_name=vector_store.collection,
                payload={"folder": new_name},
                points=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))])
            )
    folders_registry.discard(name)
    folders_registry.add(new_name)
    db_rename_folder(name, new_name)
    return {"old": name, "new": new_name}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    if doc_id not in documents_registry:
        raise HTTPException(404, f"Document {doc_id} not found")

    filename = documents_registry[doc_id].get("filename", doc_id)

    try:
        vector_store.client.delete(
            collection_name=vector_store.collection,
            points_selector=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))])
        )
    except Exception as e:
        logger.warning(f"Qdrant delete failed: {e}")

    try:
        remaining = [c for c in retriever._bm25_chunks if c.get("document_id") != doc_id]
        if remaining:
            retriever.index_chunks_for_bm25(remaining)
        else:
            retriever._bm25_index = None
            retriever._bm25_chunks = []
    except Exception as e:
        logger.warning(f"BM25 rebuild failed: {e}")

    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("DELETE FROM file_hashes WHERE doc_id = %s", (doc_id,))
        conn.commit()
        cur.close()
        conn.close()
        for h in [h for h, d in file_hashes.items() if d == doc_id]:
            del file_hashes[h]
    except Exception as e:
        logger.warning(f"DB delete failed: {e}")

    try:
        for f in UPLOAD_DIR.glob(f"{doc_id}_*"):
            f.unlink()
    except Exception as e:
        logger.warning(f"File delete failed: {e}")

    del documents_registry[doc_id]
    return {"status": "deleted", "doc_id": doc_id, "filename": filename}


@app.get("/pdf/{doc_id}/highlights")
async def get_pdf_highlights(doc_id: str, text: str = "", page: int = 1):
    """Return PyMuPDF bounding boxes for text on a given page."""
    if doc_id not in documents_registry:
        raise HTTPException(404, "Document not found")
    pdf_file = next(UPLOAD_DIR.glob(f"{doc_id}_*"), None)
    if not pdf_file:
        raise HTTPException(404, "PDF file not found on disk")

    import fitz
    doc = fitz.open(str(pdf_file))
    if page < 1 or page > len(doc):
        doc.close()
        return {"rects": [], "page_width": 0, "page_height": 0}

    pg = doc[page - 1]
    page_rect = pg.rect

    # Clean the search text
    clean = text.replace("…", "").replace("...", "").strip()
    # Try progressively shorter prefixes until hits found
    rects = []
    for length in [400, 200, 100, 60]:
        phrase = clean[:length].rsplit(" ", 1)[0] if len(clean) > length else clean
        if len(phrase) < 15:
            continue
        hits = pg.search_for(phrase, quads=False)
        if hits:
            rects = [{"x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1} for r in hits]
            break

    doc.close()
    return {"rects": rects, "page_width": page_rect.width, "page_height": page_rect.height}


@app.get("/pdf/{doc_id}")
async def get_pdf(doc_id: str):
    from fastapi.responses import FileResponse
    if doc_id not in documents_registry:
        raise HTTPException(404, "Document not found")
    for f in UPLOAD_DIR.glob(f"{doc_id}_*"):
        return FileResponse(path=str(f), media_type="application/pdf",
                          filename=documents_registry[doc_id]["filename"])
    raise HTTPException(404, "PDF file not found on disk")



@app.get("/models")
async def list_models():
    """Список доступных моделей из Ollama"""
    import httpx
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
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


@app.get("/health", dependencies=[])
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
