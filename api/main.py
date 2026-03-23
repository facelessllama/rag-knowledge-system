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
from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException, Security, Depends
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

# Minimum retrieval score to attempt an answer (0–1 cosine similarity scale).
# Below this threshold the knowledge base is considered to have no relevant content.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.30"))
MAX_CONCURRENT_QUERIES = int(os.getenv("MAX_CONCURRENT_QUERIES", "3"))
_query_semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)

parser = PDFParser(ocr_language=os.getenv("PDF_OCR_LANGUAGE", "rus+eng"))
chunker = SmartChunker(
    chunk_size=int(os.getenv("MAX_CHUNK_SIZE", "512")),
    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50"))
)
embedder = EmbeddingService(model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
vector_store = VectorStore()
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
            await retriever.index_chunks_for_bm25(bm25_chunks)

        logger.info(f"Restored: {len(documents_registry)} docs, {len(bm25_chunks)} chunks in BM25")

    except Exception as e:
        logger.warning(f"Restore failed (non-critical): {e}")

@app.on_event("startup")
async def startup():
    vector_store.create_collection(vector_size=embedder.get_vector_size())
    init_db()
    await restore_from_qdrant()
    logger.info("RAG Knowledge Base API started")


class ChatTurn(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    document_id: Optional[str] = None
    chat_history: Optional[list[ChatTurn]] = []
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


@protected.post("/upload")
async def upload_document(file: UploadFile = File(...), folder: str = Form("")):
    safe_filename = Path(file.filename).name  # strip any path components
    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    content = await file.read()
    file_hash = hashlib.md5(content).hexdigest()

    if file_hash in file_hashes:
        existing_id = file_hashes[file_hash]
        existing = documents_registry.get(existing_id, {})
        raise HTTPException(409, f"File already uploaded as '{existing.get('filename', existing_id)}' (id: {existing_id})")

    doc_id = str(uuid.uuid4())[:8]
    file_path = UPLOAD_DIR / f"{doc_id}_{safe_filename}"

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    t_parse = time.time()
    parsed = parser.parse(str(file_path))
    parse_ms = int((time.time() - t_parse) * 1000)
    chunks = chunker.chunk_document(parsed.pages, doc_id)

    if not chunks:
        raise HTTPException(422, "Could not extract text from document")

    for c in chunks:
        c.filename = safe_filename
        c.pages = parsed.total_pages
        c.folder = folder or ""

    texts = [c.text for c in chunks]
    t_embed = time.time()
    vectors = embedder.embed_batch(texts)
    embed_ms = int((time.time() - t_embed) * 1000)
    vector_store.upsert_chunks(chunks, vectors)

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

    bm25_chunks = [{"text": c.text, "chunk_id": c.chunk_id,
                    "page_num": c.page_num, "document_id": c.document_id,
                    "chunk_index": c.chunk_index, "filename": c.filename,
                    "folder": folder or ""}
                   for c in chunks]
    await retriever.index_chunks_for_bm25(bm25_chunks)

    file_hashes[file_hash] = doc_id
    try:
        with db_conn() as conn:
            conn.cursor().execute(
                "INSERT INTO file_hashes (hash, doc_id, filename) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                (file_hash, doc_id, safe_filename)
            )
    except Exception as e:
        logger.warning(f"DB save hash failed: {e}")

    if folder:
        folders_registry.add(folder)
        db_save_folder(folder)
    documents_registry[doc_id] = {
        "doc_id": doc_id,
        "filename": safe_filename,
        "pages": parsed.total_pages,
        "chunks": len(chunks),
        "size_kb": parsed.file_size_kb,
        "metadata": parsed.metadata,
        "folder": folder or ""
    }

    return {
        "doc_id": doc_id,
        "filename": safe_filename,
        "pages": parsed.total_pages,
        "chunks_created": len(chunks),
        "status": "indexed"
    }


@protected.post("/upload-batch")
async def upload_batch(files: list[UploadFile] = File(...), folder: str = Form("")):
    results = []
    for file in files:
        try:
            safe_name = Path(file.filename).name
            if not safe_name.lower().endswith(".pdf"):
                results.append({"filename": safe_name, "status": "error", "error": "Only PDF files supported"})
                continue

            content = await file.read()
            file_hash = hashlib.md5(content).hexdigest()

            if file_hash in file_hashes:
                existing_id = file_hashes[file_hash]
                existing = documents_registry.get(existing_id, {})
                results.append({"filename": safe_name, "status": "skipped", "error": f"Already uploaded as '{existing.get('filename', existing_id)}'"})
                continue

            doc_id = str(uuid.uuid4())[:8]
            file_path = UPLOAD_DIR / f"{doc_id}_{safe_name}"

            async with aiofiles.open(file_path, "wb") as f_out:
                await f_out.write(content)

            parsed = parser.parse(str(file_path))
            chunks = chunker.chunk_document(parsed.pages, doc_id)

            if not chunks:
                results.append({"filename": safe_name, "status": "error", "error": "Could not extract text"})
                continue

            for c in chunks:
                c.filename = safe_name
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
            await retriever.index_chunks_for_bm25(retriever._bm25_chunks + new_bm25)

            file_hashes[file_hash] = doc_id
            try:
                with db_conn() as conn:
                    conn.cursor().execute(
                        "INSERT INTO file_hashes (hash, doc_id, filename) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                        (file_hash, doc_id, safe_name)
                    )
            except Exception as e:
                logger.warning(f"DB save hash failed: {e}")

            documents_registry[doc_id] = {
                "doc_id": doc_id,
                "filename": safe_name,
                "pages": parsed.total_pages,
                "chunks": len(chunks),
                "size_kb": parsed.file_size_kb,
                "metadata": parsed.metadata,
                "folder": folder or ""
            }
            results.append({"doc_id": doc_id, "filename": safe_name, "status": "indexed",
                            "pages": parsed.total_pages, "chunks_created": len(chunks)})

        except Exception as e:
            logger.error(f"Batch error {file.filename}: {e}")
            results.append({"filename": file.filename, "status": "error", "error": str(e)})

    indexed = sum(1 for r in results if r["status"] == "indexed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    return {"total": len(results), "indexed": indexed, "skipped": skipped, "errors": errors, "results": results}


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
                                   chat_history=[t.model_dump() for t in request.chat_history] if request.chat_history else [],
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
                "relevance_score": round(score, 3)
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
            expansion_ms = int((time.time() - t0) * 1000)

            t1 = time.time()
            chunks = await retriever.retrieve_expanded(expanded_queries, top_k=max(20, request.top_k * 5), folder=request.folder or None)
            retrieval_ms = int((time.time() - t1) * 1000)

            scores = [c.get("score", 0) for c in chunks] if chunks else []
            score_meta = {
                "best": round(max(scores), 3) if scores else 0,
                "avg": round(sum(scores) / len(scores), 3) if scores else 0,
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

            best_score = score_meta["best"]
            if best_score < RELEVANCE_THRESHOLD:
                logger.info(f"Best score {best_score:.3f} below threshold {RELEVANCE_THRESHOLD} — not answering")
                msg = "I couldn't find relevant information in the knowledge base to answer this question."
                yield f"data: {json.dumps({'type': 'token', 'content': msg})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            t2 = time.time()
            top_chunks = reranker.rerank(request.question, chunks, top_k=request.top_k)
            rerank_ms = int((time.time() - t2) * 1000)
            reranker_type = type(reranker).__name__

            messages = prompt_builder.build(query=request.question, chunks=top_chunks,
                                            chat_history=[t.model_dump() for t in request.chat_history] if request.chat_history else [],
                                            language=request.language or None,
                                            channel=request.channel)

            t2 = time.time()
            answer_tokens = []
            async for token in generator.generate_stream(messages, model=request.model or None):
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

            # Best chunk per document, sorted by relevance
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
                        "relevance_score": round(score, 3)
                    }
            sources = sorted(seen_docs.values(), key=lambda x: x["relevance_score"], reverse=True)

            total_ms = int((time.time() - start_time) * 1000)
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'debug': {'expanded_queries': expanded_queries, 'total_ms': total_ms, 'expansion_ms': expansion_ms, 'retrieval_ms': retrieval_ms, 'rerank_ms': rerank_ms, 'generation_ms': generation_ms, 'chunks_retrieved': len(chunks), 'chunks_after_rerank': len(top_chunks), 'best_score': score_meta['best'], 'avg_score': score_meta['avg'], 'reranker': reranker_type, 'top_chunks': [{'chunk_id': c.get('chunk_id', ''), 'document_id': c.get('document_id', ''), 'score': round(c.get('rerank_score', c.get('score', 0)), 3), 'source': c.get('source', ''), 'text_preview': c.get('text', '')[:100]} for c in top_chunks]}})}\n\n"
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


@protected.patch("/documents/{doc_id}/folder")
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


@protected.get("/folders")
async def list_folders():
    # Merge folders from registry and from documents
    doc_folders = {d["folder"] for d in documents_registry.values() if d.get("folder")}
    all_folders = sorted(folders_registry | doc_folders)
    return {"folders": all_folders}


@protected.post("/folders")
async def create_folder(body: dict):
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Folder name required")
    folders_registry.add(name)
    db_save_folder(name)
    return {"name": name}


@protected.delete("/folders/{name}")
async def delete_folder(name: str):
    folders_registry.discard(name)
    db_delete_folder(name)
    return {"deleted": name}


@protected.patch("/folders/{name}")
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


@protected.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    if doc_id not in documents_registry:
        raise HTTPException(404, f"Document {doc_id} not found")

    filename = documents_registry[doc_id].get("filename", doc_id)
    errors = {}

    try:
        vector_store.client.delete(
            collection_name=vector_store.collection,
            points_selector=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))])
        )
    except Exception as e:
        logger.error(f"Qdrant delete failed for {doc_id}: {e}")
        errors["qdrant"] = str(e)

    try:
        remaining = [c for c in retriever._bm25_chunks if c.get("document_id") != doc_id]
        if remaining:
            await retriever.index_chunks_for_bm25(remaining)
        else:
            retriever._bm25_index = None
            retriever._bm25_chunks = []
    except Exception as e:
        logger.error(f"BM25 rebuild failed for {doc_id}: {e}")
        errors["bm25"] = str(e)

    try:
        with db_conn() as conn:
            conn.cursor().execute("DELETE FROM file_hashes WHERE doc_id = %s", (doc_id,))
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


@protected.get("/pdf/{doc_id}/highlights")
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
    page_width = pg.rect.width
    page_height = pg.rect.height

    if not text.strip():
        doc.close()
        return {"rects": [], "page_width": page_width, "page_height": page_height}

    # Use first ~120 chars as search anchor, cut at word boundary
    anchor = text.strip()[:120]
    last_space = anchor.rfind(' ')
    if last_space > 40:
        anchor = anchor[:last_space]

    # pg.search_for — built-in PyMuPDF text search, handles whitespace/case
    rects = pg.search_for(anchor)
    doc.close()

    return {
        "rects": [{"x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1} for r in rects],
        "page_width": page_width,
        "page_height": page_height
    }


@app.get("/pdf/{doc_id}")
async def get_pdf(doc_id: str, key: Optional[str] = None):
    """PDF открывается напрямую браузером/PDF.js — принимает ключ через query ?key="""
    from fastapi.responses import FileResponse
    if API_KEY and key != API_KEY:
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
