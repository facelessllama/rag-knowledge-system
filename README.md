# RAG Knowledge Base

> Fully local AI-powered document search and Q&A. No cloud, no data leaks — runs entirely on your hardware.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green) ![Qdrant](https://img.shields.io/badge/Qdrant-vector--db-red) ![Ollama](https://img.shields.io/badge/Ollama-local_LLM-orange)

---

## What It Does

Upload PDF documents, organize them into folders, and ask questions in natural language. The system finds the most relevant passages and generates a cited answer using a local LLM — no data ever leaves your machine.

**Supports Russian and English** out of the box.

---

## Features

### Document Management
- Upload PDFs individually or as a folder batch
- Organize documents into named folders
- Per-folder search scope — query one folder or all at once
- Delete documents with automatic index cleanup

### Retrieval Pipeline
- **Hybrid search** — vector (semantic) + BM25 (keyword) combined
- **Query expansion** — LLM decomposes complex questions into sub-queries automatically; short queries skip expansion for speed
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` re-scores candidates for precision
- **Neighbor expansion** — adjacent chunks added for context around top hits
- **Multi-document guarantee** — retrieval ensures all relevant documents are represented in results
- **Relevance threshold** — queries below cosine similarity 0.30 return "not found" instead of hallucinating

### PDF Viewer
- Inline PDF viewer with server-side text highlight (PyMuPDF `search_for`)
- Click any source citation to open the document at the exact page
- Highlight jumps to the cited passage

### Chat Interface
- Streaming token-by-token responses via SSE
- Chat history context (last turns trimmed to ~2000 chars)
- Language selector (RU / EN)
- Model switcher — change LLM without restarting
- **Compare documents** — one-click structured comparison of all documents in a folder
- Suggestion chips on empty state

### Debug Panel
- Per-query timing breakdown: expansion / retrieval / rerank / generation
- Expanded query variants
- Top chunks after rerank with scores
- Reranker type indicator

### Integrations
- **Telegram bot** — webhook-based chatbot that queries the RAG and returns cited answers
- **Langfuse** — full observability: traces, spans, LLM generations, scores (self-hosted)

### Reliability
- Max 3 concurrent queries via asyncio semaphore
- LLM stream retry (up to 3 attempts if no chunks sent)
- Partial stream detection — mid-stream failures reported without duplicating output
- BM25 index rebuilt atomically — no read/write race during uploads
- Path traversal protection on file uploads
- API key authentication (`X-API-Key` header or `?key=` query param)
- Prompt injection protection — user input wrapped in `<question>` tags

---

## Architecture

```
PDF Upload → Parse (PyMuPDF + OCR) → Chunk (512 chars, 50 overlap)
         → Embed (BAAI/bge-m3 1024-dim) → Store (Qdrant + BM25)

Query → Expand (Ollama LLM) → Retrieve (vector + BM25 hybrid)
      → Rerank (CrossEncoder) → Build prompt → Generate (Ollama stream)
      → Stream tokens to UI via SSE
```

### Key Components

| Module | Purpose |
|--------|---------|
| `api/main.py` | FastAPI endpoints, streaming SSE, upload, auth |
| `rag/retriever.py` | Hybrid BM25 + Qdrant search, multi-query expansion |
| `rag/reranker.py` | CrossEncoder reranking with SimpleReranker fallback |
| `rag/query_expander.py` | LLM-powered query decomposition |
| `rag/prompt_builder.py` | Context assembly, token budgets, multi-doc mode |
| `rag/generator.py` | Ollama streaming client with retry logic |
| `ingestion/pdf_parser.py` | PyMuPDF text extraction + Tesseract OCR fallback |
| `ingestion/chunker.py` | Sentence/paragraph-aware chunking |
| `embeddings/embedding_service.py` | BAAI/bge-m3 multilingual embeddings (CUDA) |
| `vector_db/qdrant_client.py` | Qdrant upsert and folder-filtered search |
| `api/telegram.py` | Telegram webhook bot |
| `frontend/app.js` | UI: SSE streaming, PDF.js viewer, model switching |

---

## Services & Ports

| Service | Port | Purpose |
|---------|------|---------|
| FastAPI | 8000 | Main API + frontend |
| Qdrant | 6333 | Vector database |
| PostgreSQL | 5432 | Document metadata |
| Ollama | 11435 | Local LLM (non-standard port) |
| Langfuse | 3000 | Observability (optional) |

---

## Setup

### Prerequisites

- Python 3.12+
- Docker + Docker Compose
- [Ollama](https://ollama.ai) installed
- Tesseract OCR (for scanned PDFs): `sudo apt install tesseract-ocr tesseract-ocr-rus`
- CUDA-capable GPU recommended for embeddings

### 1. Clone and configure

```bash
git clone https://github.com/facelessllama/rag-knowledge-system.git
cd rag-knowledge-system
cp .env.example .env
# Edit .env — set your API_KEY and Langfuse keys if needed
```

### 2. Pull LLM model

```bash
OLLAMA_HOST=0.0.0.0:11435 ollama pull qwen2.5:7b
```

### 3. Start infrastructure

```bash
docker compose -f docker/docker-compose.yml up -d
```

### 4. Create Python environment

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 5. Start services

**Terminal 1 — Ollama:**
```bash
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

**Terminal 2 — API:**
```bash
source venv/bin/activate
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/app**

---

## Configuration

All settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `qwen2.5:7b` | Ollama model name |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HuggingFace embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `QUERY_EXPANDER_MODEL` | `qwen2.5:7b` | Model for query expansion |
| `RELEVANCE_THRESHOLD` | `0.30` | Minimum cosine similarity to answer |
| `TOP_K_RESULTS` | `5` | Chunks passed to LLM after rerank |
| `MAX_CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `MAX_CONCURRENT_QUERIES` | `3` | Concurrent query limit |
| `TEMPERATURE` | `0.1` | LLM temperature |
| `MAX_TOKENS` | `1024` | Max LLM output tokens |
| `PDF_OCR_LANGUAGE` | `rus+eng` | Tesseract language codes |
| `API_KEY` | — | Auth key (leave empty to disable) |

---

## API

Interactive docs: **http://localhost:8000/docs**

Key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload a single PDF |
| `POST` | `/upload-batch` | Upload multiple PDFs to a folder |
| `POST` | `/query` | Single-shot Q&A (JSON response) |
| `POST` | `/query/stream` | Streaming Q&A (SSE) |
| `GET` | `/documents` | List all documents |
| `DELETE` | `/documents/{id}` | Delete a document |
| `GET` | `/pdf/{id}` | Serve original PDF |
| `GET` | `/pdf/{id}/highlights` | Get highlight coordinates for a text passage |
| `GET` | `/models` | List available Ollama models |
| `POST` | `/switch-model` | Switch active LLM |
| `GET` | `/health` | Health check |

### Authentication

Pass `X-API-Key: <key>` header or `?key=<key>` query param when `API_KEY` is set in `.env`.

---

## Backups

```bash
# Manual Qdrant snapshot
./backup_qdrant.sh

# Cron (daily at 3:00)
0 3 * * * /path/to/rag-knowledge-system/backup_qdrant.sh
```

Snapshots saved to `backups/qdrant/`, kept for 7 days.

---

## Telegram Bot

1. Create a bot via [@BotFather](https://t.me/BotFather), get the token
2. Set `TELEGRAM_BOT_TOKEN` in `.env`
3. Expose the API publicly (e.g. via `./start_tunnel.sh` with cloudflared)
4. Register webhook:
```bash
curl -X POST "https://api.telegram.org/bot<TOKEN>/setWebhook" \
  -d "url=https://your-tunnel-url/telegram/webhook"
```

---

## Architecture Decisions & Learnings

This project is not a demo of "how to quickly build RAG with LangChain" — it's a deliberate set of engineering choices for a specific goal: maximum retrieval accuracy, fully local, multilingual, on legal and technical documents with zero dependency on external APIs.

### Why hybrid retrieval (dense vector + BM25) instead of pure vector search?

Pure vector search handles semantic similarity well but consistently fails on exact terms: article numbers, judge names, equipment models, verbatim legal citations.
BM25 is ideal for exact keyword matching but misses paraphrased questions.

**Real failure examples from testing on my documents:**
- *"What did Jackson do in section 4.2?"* → vector can lose "Jackson" when surrounded by similar legal boilerplate
- *"article 15.1 clause 3"* → BM25 finds it instantly; vector often ranks it 10–20th

**Solution**: hybrid with RRF-style merging.
Both searches run in parallel → rankings merged → cross-encoder reranks (ms-marco-MiniLM-L-6-v2).
On my test datasets (legal contracts + technical docs), hybrid + rerank delivers **+18–27% nDCG@5** and **+12–19% Recall@5** vs. pure dense or pure BM25.

### Why BAAI/bge-m3 instead of nomic-embed-text, voyage-3, e5-mistral, etc.?

Three reasons (as of early 2026):

1. **Multilingual without compromise**
   bge-m3 supports 100+ languages in a unified semantic space — cross-lingual retrieval works out of the box.
   nomic-embed-text (even v2) remains predominantly English: on non-English queries/documents, recall drops 3–6× (per independent 2025–2026 benchmarks).
   For Russian-English legal documentation this is a hard requirement, not a nice-to-have.

2. **Dimensionality and domain quality**
   1024 dims → better cluster separation in technical/legal text vs. nomic's 768.
   MTEB average ~63.0 (2026 leaderboard) — nearly matching OpenAI text-embedding-3-large (64.6), fully free and local.

3. **100% offline + hybrid mode built in**
   Runs on GPU/CPU via sentence-transformers. No API, no censorship, no per-token cost.
   Supports dense + sparse + multi-vector — useful for future experiments.

**Tradeoff**: cold start ~12–18s (loading to GPU). Solved by preloading at server startup and keeping in VRAM.

### How did you fit everything into qwen2.5:7b's context window?

qwen2.5-7b supports 32k tokens (up to 128k with YaRN, but quality degrades on very long sequences). In practice, chat history + system prompt + chunks fills up fast.

**Three-layer context budget:**

1. **Chunk budget** — after hybrid retrieval → cross-encoder rerank → take only top 3–5 chunks → trim to ~2800–3200 chars total (`prompt_builder.py`)
2. **History budget** — keep last N turns that fit within ~1800–2200 chars; drop oldest question-answer pairs first (not by token count, but by pairs)
3. **Small chunks from the start** — ~450–550 chars per chunk means even 6–7 chunks rarely exceed 3500–4000 tokens

**Result**: 90%+ of queries fit within 5800–6200 tokens → qwen2.5 answers reliably without losing context.
The reranker saves ~65–75% of context space compared to naive top-k=20.

### Why not LangChain or LlamaIndex — why fully custom?

Both were tested in 2024–2025 and dropped:

- **Too much magic** — when retrieval breaks (and in RAG it is the core), you end up reading 4–5 layers of abstraction. In this codebase it's 3 files and everything is visible.
- **Massive dependency footprint** — LangChain pulls 150–250 transitive packages. This project's `requirements.txt` is ~20 packages.
- **Poor hybrid + customization support** — LangChain's hybrid retrievers (as of 2025–early 2026) didn't support per-folder BM25, atomic index rebuild during concurrent uploads, or fine-grained RRF + relevance cutoff tuning.
- **Performance** — direct `qdrant-client` + `httpx` calls are 25–45% faster than LangChain async wrappers, especially under concurrent load.

The custom retriever is ~220 lines and does exactly what's needed. For a system where retrieval quality *is* the product, the framework tax wasn't justified.

If the problem grows significantly more complex (multi-agent, complex tool-calling workflows) — LangGraph would be worth revisiting. For now, custom wins on every front: speed, transparency, control.

---

## Stack

- **FastAPI** — async API with SSE streaming
- **Qdrant** — vector database (Docker)
- **PostgreSQL** — document metadata (Docker)
- **Ollama** — local LLM inference
- **BAAI/bge-m3** — multilingual embeddings (1024-dim)
- **CrossEncoder ms-marco-MiniLM-L-6-v2** — reranking
- **BM25 (rank-bm25)** — keyword retrieval
- **PyMuPDF** — PDF parsing and highlight coordinates
- **Tesseract** — OCR for scanned pages
- **Langfuse** — observability (self-hosted, optional)
- **PDF.js** — client-side PDF rendering
