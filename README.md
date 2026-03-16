# 🔍 RAG Knowledge Base System

> Local AI-powered document search system for small business. No cloud, no data leaks — runs entirely on your hardware.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green) ![Ollama](https://img.shields.io/badge/Ollama-qwen2.5:7b-orange) ![Qdrant](https://img.shields.io/badge/Qdrant-vector--db-red)

## 🎯 What is this?

A production-ready RAG (Retrieval-Augmented Generation) system that lets you upload PDF documents and ask questions about them in natural language. Built for lawyers, clinics, and construction companies who need to keep their data local.

**Target market:** Small and medium businesses globally — law firms, clinics, construction companies.
**Languages:** Full support for Russian 🇷🇺 and English 🇬🇧 (multilingual embeddings via BAAI/bge-m3).

## ✨ Features

- 📄 **PDF Upload** — single or batch upload, automatic chunking
- 🔍 **Hybrid Search** — BM25 + vector search combined for best recall
- 🧠 **Query Expansion** — LLM decomposes complex multi-part questions automatically
- 📊 **Reranking** — cross-encoder reranks chunks for precision
- 📚 **Multi-document Reasoning** — compares and contrasts multiple documents
- 👁️ **PDF Viewer** — click any source to open PDF with highlighted text
- 🐛 **Retrieval Debug Panel** — see expanded queries, scores, latency, tokens
- 🌍 **Multilingual** — Russian, English, and 100+ languages via multilingual embeddings
- 🔒 **100% Local** — no OpenAI, no cloud, your data stays on your machine

## 🏗️ Architecture
```
User Query
    ↓
Query Expander (LLM) — splits complex questions, generates alternatives
    ↓
Hybrid Retriever — BM25 + Qdrant vector search
    ↓
Reranker (cross-encoder) — reranks top chunks
    ↓
Prompt Builder — multi-doc aware context assembly
    ↓
LLM Generator (Ollama/qwen2.5:7b)
    ↓
Answer + Sources + Debug Info
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama — qwen2.5:7b (local) |
| Embeddings | BAAI/bge-m3 (local) |
| Vector DB | Qdrant (Docker) |
| Metadata DB | PostgreSQL (Docker) |
| API | FastAPI |
| Monitoring | Langfuse |
| Frontend | Vanilla JS + PDF.js |

## 🚀 Quick Start

### Prerequisites
- Ubuntu 22.04+ / WSL2
- NVIDIA GPU (8GB+ VRAM recommended)
- Docker + Docker Compose
- Python 3.12

### 1. Clone & Setup
```bash
git clone https://github.com/YOUR_USERNAME/rag-knowledge-system
cd rag-knowledge-system
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Infrastructure
```bash
docker-compose -f docker/docker-compose.yml up -d
```

### 3. Start Ollama
```bash
OLLAMA_HOST=0.0.0.0:11435 ollama serve
ollama pull qwen2.5:7b
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your settings
```

### 5. Start API
```bash
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 6. Open UI
```
http://localhost:8000/app
```

## 📁 Project Structure
```
rag-knowledge-system/
├── api/
│   ├── main.py          # FastAPI app, endpoints
│   ├── bitrix.py        # Bitrix24 webhook integration
│   └── routes/          # Additional routes
├── rag/
│   ├── retriever.py     # Hybrid BM25 + vector retriever
│   ├── reranker.py      # Cross-encoder reranker
│   ├── generator.py     # LLM response generator
│   ├── query_expander.py # Query decomposition & expansion
│   └── prompt_builder.py # Multi-doc aware prompt builder
├── ingestion/
│   ├── pdf_parser.py    # PDF text extraction + OCR
│   └── chunker.py       # Smart text chunking
├── embeddings/
│   └── embedding_service.py  # BAAI/bge-m3 embeddings
├── frontend/
│   └── index.html       # Single-file UI with PDF viewer
├── docker/
│   └── docker-compose.yml
└── .env.example
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload single PDF |
| POST | `/upload-batch` | Upload multiple PDFs |
| POST | `/query` | Ask a question |
| GET | `/documents` | List all documents |
| DELETE | `/documents/{id}` | Delete document |
| GET | `/pdf/{doc_id}` | Serve PDF file |
| GET | `/health` | Health check |
| POST | `/bitrix/webhook` | Bitrix24 integration |

## 💡 Usage Examples

**Ask about a single document:**
```
What are the penalty terms in this contract?
```

**Multi-document comparison:**
```
Compare these documents and highlight the main differences
```

**Complex multi-part question (auto-decomposed):**
```
What is the penalty size and how much does the rent cost?
```
→ Automatically split into 2 separate searches for best results

## 📊 Performance

- Retrieval: ~500ms (hybrid search + rerank)
- Generation: ~3-8s (qwen2.5:7b local)
- Embedding: BAAI/bge-m3, 1024 dims
- Chunk size: 512 tokens, 50 overlap

## 🔧 Configuration

Key settings in `.env`:
```env
OLLAMA_URL=http://localhost:11435
QDRANT_URL=http://localhost:6333
POSTGRES_URL=postgresql://...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
```

## 📈 Roadmap

- [ ] Bitrix24 Open Line integration (webhook ready)
- [ ] Telegram bot integration
- [ ] Multi-user support
- [ ] Document versioning
- [ ] Table extraction from PDFs

## 👨‍💻 Author

Built as a portfolio project demonstrating production-ready RAG architecture.

**Target industries:** Law firms, clinics, construction companies

**Languages supported:** Russian, English, and 100+ languages via BAAI/bge-m3 multilingual embeddings.
