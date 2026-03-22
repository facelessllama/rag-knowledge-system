# RAG Knowledge Base System

> Local AI-powered document search. No cloud, no data leaks — runs entirely on your hardware.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green) ![Ollama](https://img.shields.io/badge/Ollama-qwen2.5:7b-orange) ![Qdrant](https://img.shields.io/badge/Qdrant-vector--db-red)

## What is this?

A production-ready RAG (Retrieval-Augmented Generation) system — upload PDF documents, ask questions in natural language, get answers with source citations and PDF highlighting. Built to keep data local.

**Languages:** Russian and English (multilingual embeddings via BAAI/bge-m3).

## Features

- **PDF Upload** — single, batch, or entire folder upload with drag-and-drop
- **Folder Organisation** — group documents into folders, filter search by folder
- **Hybrid Search** — BM25 + vector search combined, with relevance threshold (no hallucinated answers when nothing relevant is found)
- **Query Expansion** — LLM decomposes complex multi-part questions automatically
- **Cross-encoder Reranking** — `ms-marco-MiniLM-L-6-v2` reranks candidates for precision
- **Streaming Responses** — token-by-token SSE streaming
- **PDF Viewer** — click any source to open PDF with highlighted passage
- **Chat History** — conversation context across follow-up questions
- **Retrieval Debug Panel** — expanded queries, scores, latency, token count
- **Multi-document Reasoning** — compares and contrasts multiple documents
- **Model Switching** — switch LLM per-request from the UI
- **Telegram Bot** — webhook integration: any message to the bot gets answered by RAG

## Architecture

```
User Query
    ↓
Query Expander (LLM) — splits complex questions, generates alternatives
    ↓
Hybrid Retriever — BM25 + Qdrant vector search, folder-filtered
    ↓
Relevance Threshold — returns "not found" if best score < 0.30
    ↓
Cross-encoder Reranker (ms-marco-MiniLM-L-6-v2)
    ↓
Prompt Builder — context budget guard, multi-doc aware
    ↓
LLM Generator (Ollama / qwen2.5:7b) — streaming
    ↓
Answer + Sources + Debug Info
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama — qwen2.5:7b (local) |
| Embeddings | BAAI/bge-m3 (local, 1024-dim) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector DB | Qdrant (Docker) |
| Metadata DB | PostgreSQL (Docker) |
| API | FastAPI + SSE streaming |
| Monitoring | Langfuse (optional) |
| Frontend | Vanilla JS, PDF.js |

## Quick Start

### Prerequisites
- Ubuntu 22.04+ / WSL2
- NVIDIA GPU (8GB+ VRAM recommended)
- Docker + Docker Compose
- Python 3.12
- Ollama installed

### 1. Clone & install

```bash
git clone https://github.com/facelessllama/rag-knowledge-system
cd rag-knowledge-system
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Start infrastructure

```bash
docker compose -f docker/docker-compose.yml up -d
```

### 4. Start Ollama

```bash
OLLAMA_HOST=0.0.0.0:11435 ollama serve
ollama pull qwen2.5:7b
```

### 5. Start API

```bash
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 6. Open UI

```
http://localhost:8000/app
```

## Telegram Bot Integration

Demonstrates how any external system can integrate with RAG via a simple webhook.

```bash
# 1. Create a bot via @BotFather, get token
# 2. Add to .env
echo "TELEGRAM_BOT_TOKEN=your-token" >> .env

# 3. Expose local server (ngrok)
ngrok http 8000

# 4. Register webhook
curl -X POST "https://api.telegram.org/bot<TOKEN>/setWebhook" \
     -d "url=https://<NGROK_URL>/telegram/webhook"
```

Any message sent to the bot is answered by the RAG system with source citations. The same pattern applies to any other platform (Slack, WhatsApp, custom CRM).

## Project Structure

```
rag-knowledge-system/
├── api/
│   ├── main.py              # FastAPI app, all endpoints
│   └── telegram.py          # Telegram bot webhook
├── rag/
│   ├── retriever.py         # Hybrid BM25 + vector retriever
│   ├── reranker.py          # Cross-encoder + SimpleReranker fallback
│   ├── generator.py         # LLM streaming generator
│   ├── query_expander.py    # Query decomposition & expansion
│   └── prompt_builder.py    # Context budget guard, multi-doc prompts
├── ingestion/
│   ├── pdf_parser.py        # PDF text extraction + OCR fallback
│   └── chunker.py           # Sentence-boundary text chunking
├── embeddings/
│   └── embedding_service.py # BAAI/bge-m3 embeddings
├── vector_db/
│   └── qdrant_client.py     # Qdrant CRUD, folder-filtered search
├── frontend/                # No build step — plain HTML/JS/CSS
│   ├── index.html           # HTML structure and layout only
│   ├── app.js               # UI logic, streaming, PDF viewer
│   ├── api.js               # API client functions
│   └── styles.css           # Styles
├── docker/
│   └── docker-compose.yml   # Qdrant + PostgreSQL + Langfuse
├── .env.example
└── requirements.txt
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload single PDF |
| POST | `/upload-batch` | Upload multiple PDFs |
| POST | `/query` | Ask a question |
| POST | `/query/stream` | Ask a question (SSE streaming) |
| GET | `/documents` | List documents and folders |
| DELETE | `/documents/{id}` | Delete document |
| PATCH | `/documents/{id}/folder` | Move document to folder |
| GET | `/folders` | List all folders |
| POST | `/folders` | Create folder |
| DELETE | `/folders/{name}` | Delete folder |
| PATCH | `/folders/{name}` | Rename folder |
| GET | `/pdf/{doc_id}` | Serve PDF file |
| GET | `/pdf/{doc_id}/highlights` | Get highlight coordinates |
| GET | `/models` | List available Ollama models |
| GET | `/health` | Health check |
| POST | `/telegram/webhook` | Telegram bot webhook |

## Configuration

`.env` key settings:

```env
OLLAMA_URL=http://localhost:11435   # non-default port to avoid conflicts
QDRANT_URL=http://localhost:6333
POSTGRES_URL=postgresql://raguser:ragpass@localhost:5432/ragdb
LANGFUSE_PUBLIC_KEY=...             # optional
LANGFUSE_SECRET_KEY=...             # optional
TELEGRAM_BOT_TOKEN=...              # optional, Telegram bot integration
```

## Performance

- Retrieval: ~200–500ms (hybrid search + cross-encoder rerank)
- Generation: ~3–8s first token (qwen2.5:7b local)
- Streaming: token-by-token via SSE
- Embeddings: BAAI/bge-m3, 1024 dims, CUDA accelerated

## Author

Built as a portfolio project demonstrating production-ready RAG architecture.
