"""
RAG Knowledge Base API
"""
import logging
import uuid
import time
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import aiofiles
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor

from langfuse import Langfuse

from ingestion.pdf_parser import PDFParser
from ingestion.chunker import SmartChunker
from embeddings.embedding_service import EmbeddingService
from vector_db.qdrant_client import VectorStore
from rag.retriever import HybridRetriever
from rag.reranker import SimpleReranker
from rag.prompt_builder import PromptBuilder
from rag.generator import LLMGenerator
from rag.query_expander import QueryExpander

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Knowledge Base API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

parser = PDFParser()
chunker = SmartChunker(chunk_size=512, chunk_overlap=50)
embedder = EmbeddingService()
vector_store = VectorStore()
retriever = HybridRetriever(embedder, vector_store)
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

def get_db():
    return psycopg2.connect(os.getenv("POSTGRES_URL", "postgresql://raguser:ragpass@localhost:5432/ragdb"))

def init_db():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS file_hashes (
                hash VARCHAR(32) PRIMARY KEY,
                doc_id VARCHAR(8) NOT NULL,
                filename VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()
        cur.execute("SELECT hash, doc_id FROM file_hashes")
        for row in cur.fetchall():
            file_hashes[row[0]] = row[1]
        cur.close()
        conn.close()
        logger.info(f"DB ready | loaded {len(file_hashes)} file hashes")
    except Exception as e:
        logger.warning(f"DB init failed: {e}")

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

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    model: str
    tokens_used: int


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
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

    texts = [c.text for c in chunks]
    vectors = embedder.embed_batch(texts)
    vector_store.upsert_chunks(chunks, vectors)

    bm25_chunks = [{"text": c.text, "chunk_id": c.chunk_id,
                    "page_num": c.page_num, "document_id": c.document_id,
                    "chunk_index": c.chunk_index, "filename": c.filename}
                   for c in chunks]
    retriever.index_chunks_for_bm25(bm25_chunks)

    file_hashes[file_hash] = doc_id
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO file_hashes (hash, doc_id, filename) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
            (file_hash, doc_id, file.filename)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"DB save hash failed: {e}")

    documents_registry[doc_id] = {
        "doc_id": doc_id,
        "filename": file.filename,
        "pages": parsed.total_pages,
        "chunks": len(chunks),
        "size_kb": parsed.file_size_kb,
        "metadata": parsed.metadata
    }

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "pages": parsed.total_pages,
        "chunks_created": len(chunks),
        "status": "indexed"
    }


@app.post("/upload-batch")
async def upload_batch(files: list[UploadFile] = File(...)):
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

            texts = [c.text for c in chunks]
            vectors = embedder.embed_batch(texts)
            vector_store.upsert_chunks(chunks, vectors)

            new_bm25 = [{"text": c.text, "chunk_id": c.chunk_id,
                         "page_num": c.page_num, "document_id": c.document_id,
                         "chunk_index": c.chunk_index, "filename": c.filename}
                        for c in chunks]
            retriever.index_chunks_for_bm25(retriever._bm25_chunks + new_bm25)

            file_hashes[file_hash] = doc_id
            try:
                conn = get_db()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO file_hashes (hash, doc_id, filename) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                    (file_hash, doc_id, file.filename)
                )
                conn.commit()
                cur.close()
                conn.close()
            except Exception as e:
                logger.warning(f"DB save hash failed: {e}")

            documents_registry[doc_id] = {
                "doc_id": doc_id,
                "filename": file.filename,
                "pages": parsed.total_pages,
                "chunks": len(chunks),
                "size_kb": parsed.file_size_kb,
                "metadata": parsed.metadata
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

    start_time = time.time()

    trace = None
    if LANGFUSE_ENABLED:
        try:
            trace = langfuse.trace(name="rag_query", input=request.question, tags=["query"])
        except Exception as e:
            logger.warning(f"Langfuse trace failed: {e}")

    t0 = time.time()
    expanded_queries = await query_expander.expand(request.question)
    chunks = await retriever.retrieve_expanded(expanded_queries, top_k=request.top_k * 2)
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

    top_chunks = reranker.rerank(request.question, chunks, top_k=min(request.top_k, 3))

    messages = prompt_builder.build(query=request.question, chunks=top_chunks,
                                   chat_history=request.chat_history)

    t1 = time.time()
    result = await generator.generate(messages)
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
                        model=result["model"], tokens_used=result["total_tokens"])


@app.get("/documents")
async def list_documents():
    return {"total": len(documents_registry), "documents": list(documents_registry.values())}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    if doc_id not in documents_registry:
        raise HTTPException(404, f"Document {doc_id} not found")

    filename = documents_registry[doc_id].get("filename", doc_id)

    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
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


@app.get("/pdf/{doc_id}")
async def get_pdf(doc_id: str):
    from fastapi.responses import FileResponse
    if doc_id not in documents_registry:
        raise HTTPException(404, "Document not found")
    for f in UPLOAD_DIR.glob(f"{doc_id}_*"):
        return FileResponse(path=str(f), media_type="application/pdf",
                          filename=documents_registry[doc_id]["filename"])
    raise HTTPException(404, "PDF file not found on disk")


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
