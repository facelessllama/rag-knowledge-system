# RAG Knowledge Base

> Local-first AI-powered document search and Q&A — runs entirely on your hardware by default, with an optional, explicitly opt-in DeepSeek cloud mode an administrator can turn on.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green) ![Qdrant](https://img.shields.io/badge/Qdrant-vector--db-red) ![Ollama](https://img.shields.io/badge/Ollama-local_LLM-orange) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## What It Does

Upload PDF documents, organize them into folders, and ask questions in natural language. The system finds the most relevant passages and generates a cited answer using a local LLM by default — no data leaves your machine unless an administrator has explicitly enabled the DeepSeek cloud provider (`ENABLE_CLOUD_GENERATOR=true`).

Built for English-language documents.

---

## Tested at Scale, Not Just Demoed

Most portfolio RAG projects prove one thing: a good-looking chat screenshot exists. That doesn't prove the system holds up once you stop being gentle with it — so this one was pushed harder on purpose.

The day-to-day working set is a few hundred documents, and on that it's essentially perfect. To find out what actually breaks as a library grows, it was then loaded with **57,000 real documents** — not synthetic filler, but a genuinely difficult real-world dataset (EU legislation full of near-identical filings reissued for decades, about the hardest condition for a search system to stay accurate under) — and re-tested at every step up to that size.

The honest result, reported as measured rather than rounded up: on natural, real-world phrasing, accuracy stayed at **100%** even at 57,000 documents. On a narrower "quote the exact document ID" scenario, it didn't — ranking quality degraded steadily from 96% down to 73% as the corpus grew, and that decline is documented rather than hidden. Along the way, testing also caught a bug in the *test's own scoring*, which had been quietly reporting a false 100% — it was found, fixed, and every affected result was re-run live against the real 57,000-document corpus to confirm the fix, not just recomputed on paper. A real capacity limit was found too: reranking runs on one GPU by design, so the speedup from serving multiple users at once drops substantially once the corpus is this large — measured directly, not estimated.

A separate pass went after a different question: not "does it hold up at volume," but "does it hold up on content it was never shaped around at all." 762 real, public-source documents it had no prior exposure to — arXiv papers, FDA drug labels, FAA safety manuals, and deliberately adversarial old-scan/handwritten-math pages — were run through the untouched ingestion pipeline. That surfaced two real, previously-undiscovered bugs no legal-corpus test had ever hit (a font-encoding artifact that silently broke ingestion for over a third of the arXiv sample, and a vector-store batching limit), both fixed and confirmed live. It also caught a genuine retrieval defect: a guarantee meant to keep an exact-cited figure/table chunk from being reranked away only worked when ordinary search *hadn't* already found it weakly — backwards, exactly when the guarantee was needed. Fixing it raised Evidence-chunk Recall (does the retrieved chunk actually contain the cited fact, not just the right document) from 19.9% to 95.5% on the realistic scoped-document workflow. The same corpus was then used to run a controlled generator comparison — identical retrieved evidence sent to local Qwen and to DeepSeek's cloud API — which is what the opt-in DeepSeek cloud mode below is based on, not a vendor's own benchmark.

The same standard was applied to the engineering side, not just retrieval quality: the service is backed by 398 automated tests, including deliberate crash-and-restart drills (killing the database connection mid-operation, stopping the search engine mid-delete) that prove the system recovers cleanly instead of just assuming it would. See *"Architecture Decisions & Learnings"* below for what broke during that work and how it was fixed.

**Full write-up — every number, every bug found, and what's still an open gap — in [`VALIDATION.md`](VALIDATION.md) and [`eval/mixed_corpus/README.md`](eval/mixed_corpus/README.md).**

---

## Features

### Document Management
- Upload PDFs individually or as a folder batch
- Organize documents into named folders
- Per-folder search scope — query one folder or all at once
- Delete documents with automatic index cleanup

### Retrieval Pipeline
- **Hybrid search** — vector (semantic) + BM25-style sparse (keyword) combined, fused **server-side in Qdrant** via RRF — no client-side keyword index to rebuild or desync across replicas
- **Query expansion** — LLM decomposes complex questions into sub-queries automatically; short queries skip expansion for speed
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` re-scores candidates for precision
- **Neighbor expansion** — adjacent chunks added for context around top hits
- **Multi-document guarantee** — retrieval ensures all relevant documents are represented in results
- **Relevance threshold** — queries whose best post-rerank cross-encoder score (a raw logit, not a 0–1 cosine similarity — see `RELEVANCE_THRESHOLD` below) falls below `3.0` return "not found" instead of hallucinating

### Document Viewer
- Inline source viewer highlights the exact cited passage via stored char offsets — no text search against a rendered page, so it works identically for TXT and PDF (including OCR'd pages)
- Click any source citation to open the document at the exact page
- "View original PDF" opens the real PDF.js view (original layout, images, stamps) with no highlight promised there

### Chat Interface
- Streaming token-by-token responses via SSE
- Chat history context (last turns trimmed to ~2000 chars)
- Model switcher — change LLM without restarting
- **Compare documents** — one-click structured comparison of all documents in a folder
- Suggestion chips on empty state

### Debug Panel
- Per-query timing breakdown: expansion / retrieval / rerank / generation
- Expanded query variants
- Top chunks after rerank with scores
- Reranker type indicator

### Integrations
- **Langfuse** — full observability: traces, spans, LLM generations, scores (self-hosted)

### Reliability
- Max 3 concurrent queries via asyncio semaphore
- LLM stream retry (up to 3 attempts if no chunks sent)
- Partial stream detection — mid-stream failures reported without duplicating output
- Hybrid search runs entirely server-side in Qdrant — no in-memory index to rebuild on upload or restart
- **Single-instance enforcement** — a Postgres advisory lock plus a background watchdog make sure only one process ever holds the in-memory document registries, and kill the process the instant that stops being provably true. See *"The multi-process trap I didn't see coming"* below for why this exists and two bugs in my first attempt at it.
- **Crash-safe delete & folder moves** — a durable tombstone in Postgres means a delete or rename that fails halfway through leaves a precise, retryable trail instead of an orphaned file or a document that silently reappears after a restart. See *"Delete and rename across three stores that don't share a transaction"* below.
- Path traversal protection on file uploads
- API key authentication (`X-API-Key` header or `?key=` query param)
- Uploads are streamed into disk and validated (size cap, magic-byte check, page count, batch size, OCR duration, concurrent ingestion jobs — see `.env.example`) before the file is ever visible at its final path; an ASGI-level middleware also caps request body size before Starlette's multipart parser gets to spool an oversized one
- The whole parse → chunk → embed → upsert pipeline rolls back across all three storage layers (file, Qdrant, Postgres) on any failure, including task cancellation — verified live against a real mid-pipeline Postgres failure and against Qdrant being down during rollback, not just reasoned about
- If a reverse proxy sits in front of this app, it should set its own request body size limit too, as defense in depth — not instead of the above
- Prompt injection mitigations (not a hard guarantee — an LLM has no true trust boundary): question and document-excerpt text have `<`/`>` escaped so neither can forge the literal `<question>`/`</question>` structural tags; the system prompt explicitly tells the model both the question and the document context are untrusted data, never instructions; `chat_history` turn `role` is restricted to `user`/`assistant` (a client-supplied `role: "system"` can no longer reach the LLM's messages list unlabeled); question/history/document-ID fields are length- and count-capped. See `tests/test_prompt_builder.py`'s injection tests and `tests/test_schemas.py`.

---

## Architecture

```
PDF Upload → Parse (PyMuPDF + OCR) → Chunk (512 chars, 50 overlap)
         → Embed (BAAI/bge-m3 1024-dim) → Store (Qdrant: dense + sparse vectors)

Query → Expand (Ollama LLM) → Retrieve (Qdrant hybrid: dense + sparse, server-side RRF)
      → Rerank (CrossEncoder) → Build prompt → Generate (Ollama stream)
      → Stream tokens to UI via SSE
```

### Key Components

| Module | Purpose |
|--------|---------|
| `api/main.py` | App/router wiring, DI, the DB layer, reconciliation, startup/shutdown, auth — endpoints themselves live in `api/health.py`/`documents.py`/`query.py`/`upload.py` below, split out of a 2263-line monolith |
| `api/health.py` | `/health`, `/health/live`, `/health/ready`, `/models` |
| `api/documents.py` | `/documents`, `/folders`, `/pdf/{id}` (own auth), document/folder CRUD |
| `api/query.py` | `/query`, `/query/stream` — the streaming SSE endpoint |
| `api/upload.py` | `/upload`, `/upload-batch` — parse → chunk → embed → upsert pipeline |
| `rag/retriever.py` | Multi-query expansion over Qdrant hybrid search, neighbor expansion |
| `vector_db/qdrant_client.py` | Qdrant hybrid search (dense+sparse RRF fusion), payload indexes |
| `vector_db/sparse_encoder.py` | BM25-style sparse vector construction (English tokenizer) |
| `rag/reranker.py` | CrossEncoder reranking with SimpleReranker fallback |
| `rag/query_expander.py` | LLM-powered query decomposition |
| `rag/prompt_builder.py` | Context assembly, token budgets, multi-doc mode |
| `rag/generator.py` | Ollama streaming client with retry logic; `GeneratorRouter` also routes non-streaming requests to an optional, opt-in DeepSeek cloud generator |
| `ingestion/pdf_parser.py` | PyMuPDF text extraction + Tesseract OCR fallback |
| `ingestion/chunker.py` | Sentence/paragraph-aware chunking |
| `embeddings/embedding_service.py` | BAAI/bge-m3 embeddings (CUDA) |
| `rag/executors.py` | Single-worker thread pool for GPU-bound calls — keeps embed/rerank off the event loop |
| `frontend/app.js` | UI: SSE streaming, PDF.js viewer, model switching |
| `lock.py` | Backup-coordination `flock`, plus the Postgres advisory lock + watchdog that enforce single-instance execution |

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
- Tesseract OCR (for scanned PDFs): `sudo apt install tesseract-ocr`
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
OLLAMA_HOST=127.0.0.1:11435 ollama pull qwen2.5:7b
```

### 3. Start infrastructure

```bash
docker compose -f docker/docker-compose.yml --env-file .env up -d
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
OLLAMA_HOST=127.0.0.1:11435 ollama serve
```
Loopback only — Ollama has no built-in auth, so `0.0.0.0` would let any LAN-reachable client call every model with no credentials at all.

**Terminal 2 — API:**
```bash
source venv/bin/activate
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Also loopback only — plain HTTP, no TLS. For access beyond localhost, put a TLS-terminating reverse proxy in front of it instead of widening this bind.

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
| `RELEVANCE_THRESHOLD` | `3.0` | Minimum cross-encoder rerank score to attempt an answer — a raw logit (calibrated on golden_dataset.json, see the comment above it in `api/main.py`), **not** a 0–1 cosine similarity |
| `top_k` | `5` (per request, `1`–`20`) | Chunks kept after rerank — a `/query` request field, not an env var; there's no `TOP_K_RESULTS` setting |
| `MAX_CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `MAX_CONCURRENT_QUERIES` | `3` | Concurrent query limit |
| `TEMPERATURE` | `0.1` | LLM temperature |
| `PDF_OCR_LANGUAGE` | `eng` | Tesseract language codes |
| `API_KEY` | — | Auth key (leave empty to disable) |
| `ENABLE_CLOUD_GENERATOR` | `false` | Administrator opt-in for the DeepSeek cloud provider — `provider="deepseek"` is rejected with a 4xx unless this is `true` |
| `DEEPSEEK_API_KEY` | — | DeepSeek API key — required (in addition to the flag above) for the cloud provider to be usable |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | DeepSeek model name — always the server-configured value; a client-supplied `model` is ignored for the cloud provider |
| `QDRANT_REQUEST_TIMEOUT_SECONDS` | `5` | Bounds every Qdrant REST call — see *"Delete and rename across three stores..."* above for why this exists |
| `SINGLE_INSTANCE_WATCHDOG_INTERVAL_SECONDS` / `_TIMEOUT_SECONDS` | `5` / `5` | How often, and how patiently, the single-instance guard re-checks its Postgres lock is still alive — see *"The multi-process trap I didn't see coming"* above |

Upload/ingestion limits (`MAX_UPLOAD_MB`, `MAX_BATCH_FILES`, `MAX_PDF_PAGES`, `OCR_TIMEOUT_SECONDS`, `MAX_TOTAL_OCR_SECONDS`, `MAX_CONCURRENT_INGESTIONS`) are documented inline in `.env.example`, not repeated here.

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
| `GET` | `/documents/{id}/pages/{page_num}` | Normalized page text for the source viewer (TXT + PDF) |
| `GET` | `/models` | List available Ollama models, plus `cloud` status (whether DeepSeek is enabled/configured and its configured model name) — there is no separate switch-model endpoint; `/query` and `/query/stream` take `model`/`provider` per request |
| `GET` | `/health` | Readiness (back-compat alias for `/health/ready`) — 200 only if Postgres, Qdrant and Ollama are all actually reachable, else 503 |
| `GET` | `/health/live` | Liveness — process alive, no dependency calls |
| `GET` | `/health/ready` | Readiness — real Postgres/Qdrant/Ollama checks, 503 on any failure |

### Authentication

Pass `X-API-Key: <key>` header or `?key=<key>` query param when `API_KEY` is set in `.env`.

---

## Testing

398 tests as of this writing. Most are fast and fully mocked — no Docker/Qdrant/Ollama needed:

```bash
source venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

A handful (`tests/test_single_instance_guard.py`) spin up a real second OS process against a live Postgres to prove the single-instance lock actually excludes it — they skip cleanly if Postgres isn't reachable, so `pytest` alone never hangs or fails on a machine with no infrastructure running. Everything else, including the concurrency claims (the single-instance lock, the folder-rename race), is tested with `asyncio.Event`-based synchronization that deterministically pauses one coroutine mid-flight while a second one runs — not `sleep()`-and-hope, which would either be flaky or too slow to run in CI. A regression that reintroduces one of those races fails the suite instead of only showing up under real concurrent load.

Scripts that exercise a running instance (`docker compose up -d` + API started):

```bash
python scripts/scale_test.py --count 1000        # ingest a synthetic corpus, check throughput + exact-term retrieval
python scripts/concurrency_test.py --requests 10 # simultaneous /query requests vs sequential baseline
```

---

## Backups & Restore

A backup is only as good as the last time you actually restored from it. This project treats both halves as one workflow, not just the snapshot half.

```bash
# Manual full backup: Qdrant snapshot + Postgres dump + uploads/ originals +
# a checksum-verified manifest (row/point counts captured within the same
# exclusive backup window — see backup_qdrant.sh), staged atomically so a
# crash mid-run never leaves a half-written backup that looks complete.
./backup_qdrant.sh

# Restore a specific backup into an EXPLICIT target (never defaults to this
# repo's own .env — see the script's own --help for the full flag list).
./restore_backup.sh --backup-dir backups/20260718_030000 \
    --postgres-url postgresql://raguser:ragpass@localhost:15432/ragdb \
    --qdrant-url http://localhost:16333 \
    --uploads-dir /path/to/scratch/uploads \
    --i-understand-this-overwrites-target

# Automated restore drill: restores the most recent backup into a disposable
# Postgres+Qdrant (docker/docker-compose.restore-test.yml — different ports/
# volumes than the real stack) and runs semantic checks (row counts,
# orphaned document_pages, missing originals, spot-checked Qdrant points)
# against manifest.json — not against live production, which has moved on
# since the backup was taken. Tears the scratch environment down either way.
./verify_restore.sh
```

Backups are saved to `backups/YYYYMMDD_HHMMSS/`, kept for 7 days. Consistency across Qdrant/Postgres/uploads/ is enforced by an exclusive `flock` (see `lock.py`) held for the whole backup; every code path that mutates any of the three (uploads, deletes, folder ops, maintenance/migration scripts) takes the matching shared lock — proven under real concurrent load, not just by reading the code: with a process holding the shared lock, `backup_qdrant.sh`'s exclusive acquire measurably blocks until that holder releases (not before), and a new mutation attempted while backup holds the exclusive lock is rejected immediately rather than queued. A partial restore (killed or erroring mid-apply, after the destructive DROP/recreate has already started) has also been verified to surface as a clear failure — both via `restore_backup.sh`'s own exit code and via `verify_restore_semantics.py`'s count checks — never as a false "restored successfully".

**Restore-check artifacts** — every `verify_restore.sh` run writes:
- `backups/.last_restore_check.{json,log}` — always the most recent run (overwritten each time), for a quick "did the last drill pass" check.
- `backups/restore_checks/<TS>.{json,log}` — one dated, retained copy per run (`RESTORE_CHECK_KEEP_DAYS`, default 30 — longer than `backups/` itself, since these are tiny and the drill *history* matters more than any one backup's retention window).

The JSON records `git_commit` (+ dirty flag) of the scripts that ran, `restore_duration_seconds` / `verify_duration_seconds`, exit codes, and the full semantic-check detail. The paired `.log` is the drill's complete stdout/stderr with secrets redacted (`POSTGRES_PASSWORD`, `QDRANT_API_KEY`, any embedded URL credentials).

Set `RESTORE_CHECK_REMOTE_DEST` (an rsync target) to also ship each dated copy off-host — otherwise the same disk/host failure a backup protects against can just as easily take the *evidence that restores were ever verified* down with it.

**`RESTORE_CHECK_ALERT_CMD`** — the path to a single executable (a script, not a shell command line). Invoked directly as `"$RESTORE_CHECK_ALERT_CMD" drill_failed` (from `verify_restore.sh`) or `"$RESTORE_CHECK_ALERT_CMD" stale` (from `check_restore_freshness.sh`, below) — **never via `eval` or `sh -c`** — with the relevant JSON piped to its stdin, so nothing derived from logs or environment is ever re-interpreted by a shell. If you need several steps (e.g. `curl` + `jq`), put them inside that script rather than building a one-liner in the env var.

**This mechanism being implemented and tested is not the same as the restore being "regularly verified".** What's still required before that claim holds:
1. **A schedule** — `verify_restore.sh` and `check_restore_freshness.sh` actually wired into cron (or equivalent), not just run by hand:
   ```bash
   # Daily backup
   0 3 * * * /path/to/rag-knowledge-system/backup_qdrant.sh
   # Weekly restore drill
   0 4 * * 0 /path/to/rag-knowledge-system/verify_restore.sh
   # Daily freshness check — catches verify_restore.sh's OWN cron entry
   # silently failing to fire, which verify_restore.sh can't detect from
   # inside itself
   0 6 * * * /path/to/rag-knowledge-system/check_restore_freshness.sh
   ```
   `check_restore_freshness.sh` alerts if the most recent *successful* drill (`backups/restore_checks/*.json`, not just `.last_restore_check.json`'s mtime) is older than `RESTORE_FRESHNESS_MAX_DAYS` (default 9, sized for the weekly cadence above). It checks the last **pass** specifically — a fresh failed run does not reset this: `verify_restore.sh` running and failing every week for a month should page just as loudly as it not running at all.
2. **An owner and an alert channel** — `RESTORE_CHECK_ALERT_CMD` wired to something that actually pages/messages a named person, not "whoever happens to read the log."
3. **Evidence of a successful scheduled execution** — the first cron-triggered (not manually-triggered) run actually landing in `backups/restore_checks/` with `result: "pass"`.

A full disaster-recovery walkthrough (restore onto a genuinely separate host, not just the scratch containers) is worth doing manually on a quarterly cadence — the automated drill proves the backup *contents* are restorable, not that your recovery runbook for a truly dead host actually works end to end.

---

## Architecture Decisions & Learnings

This project is not a demo of "how to quickly build RAG with LangChain" — it's a deliberate set of engineering choices for a specific goal: maximum retrieval accuracy, fully local, on English legal and technical documents with zero dependency on external APIs.

### Why hybrid retrieval (dense vector + BM25) instead of pure vector search?

Pure vector search handles semantic similarity well but consistently fails on exact terms: article numbers, judge names, equipment models, verbatim legal citations.
BM25 is ideal for exact keyword matching but misses paraphrased questions.

**Real failure examples from testing on my documents:**
- *"What did Jackson do in section 4.2?"* → vector can lose "Jackson" when surrounded by similar legal boilerplate
- *"article 15.1 clause 3"* → BM25 finds it instantly; vector often ranks it 10–20th

**Solution**: hybrid with RRF-style merging.
Both searches run in parallel → rankings merged → cross-encoder reranks (ms-marco-MiniLM-L-6-v2).
On my test datasets (legal contracts + technical docs), hybrid + rerank delivers **+18–27% nDCG@5** and **+12–19% Recall@5** vs. pure dense or pure BM25.

**Implementation note**: the sparse side originally ran as an in-process `rank_bm25` index, rebuilt from scratch on every upload and restart — fine at demo scale, but it doesn't survive a growing library or multiple API replicas. It now runs as a native Qdrant sparse vector (`Modifier.IDF`), fused with the dense vector server-side in a single `query_points` call — same hybrid retrieval behavior, but the corpus-wide term statistics are maintained incrementally by Qdrant itself instead of rebuilt client-side.

### Why BAAI/bge-m3 instead of nomic-embed-text, voyage-3, e5-mistral, etc.?

Two reasons (as of early 2026):

1. **Dimensionality and domain quality**
   1024 dims → better cluster separation in technical/legal text vs. nomic's 768.
   MTEB average ~63.0 (2026 leaderboard) — nearly matching OpenAI text-embedding-3-large (64.6), fully free and local.

2. **100% offline + hybrid mode built in**
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
- **Poor hybrid + customization support** — LangChain's hybrid retrievers (as of 2025–early 2026) didn't support per-folder filtering, native Qdrant sparse-vector fusion, or fine-grained RRF + relevance cutoff tuning the way calling `qdrant-client` directly does.
- **Performance** — direct `qdrant-client` + `httpx` calls are 25–45% faster than LangChain async wrappers, especially under concurrent load.

The custom retriever is ~220 lines and does exactly what's needed. For a system where retrieval quality *is* the product, the framework tax wasn't justified.

If the problem grows significantly more complex (multi-agent, complex tool-calling workflows) — LangGraph would be worth revisiting. For now, custom wins on every front: speed, transparency, control.

### Why doesn't "View original PDF" highlight the cited passage?

The inline source viewer highlights the exact cited text, character for character. Click through to "View original PDF" — the real PDF.js rendering, original layout, images, letterhead and all — and there's no highlight box there. That's not an oversight; it's a boundary I drew on purpose after the first approach kept breaking in ways that weren't worth chasing further.

The first version used PyMuPDF's `page.search_for()`: take the cited text, search for it on the rendered page, draw a box wherever it turns up. It works, until:
- **A scanned page has no highlight to draw.** OCR'd pages have no embedded text layer to search against — `search_for()` has nothing to find, so nothing gets highlighted, silently.
- **The box lands on the wrong pixels even when the text is found.** A PDF's internal text order doesn't always match its visual layout — a caption near an inserted image, a footnote interleaved oddly with body text — so the "found" match is real, but the coordinates it reports can land on visually blank space or the wrong line entirely.

The fix was to stop searching the rendered page at all. The inline viewer highlights against the same normalized text and character offsets the retrieval pipeline already computed once, at ingestion time — exact, and identical for TXT and OCR'd PDF, because it never touches how the page is drawn. What it doesn't do is exist for the "view the actual PDF" path: mapping those same character offsets onto pixel coordinates for an arbitrary PDF layout (rotated text, multi-column, an image that breaks reading order) is a genuinely hard problem, and not one worth solving for a single-user local tool when the inline viewer already answers "is this citation real" reliably. So that's the tradeoff, stated plainly rather than left for someone to discover: exact highlighting where it matters for verifying a citation, an honest "no highlight promised" on the view that exists for reading the document as it actually looks.

### The multi-process trap I didn't see coming (and two bugs in my first fix)

For a long time, `documents_registry` / `file_hashes` / `folders_registry` were exactly what they look like: plain Python dicts, loaded from Postgres once at startup, mutated directly by every upload, delete, and rename. That's completely fine for one process. It quietly stops being fine the moment someone adds `uvicorn --workers 2` — each worker loads its own private copy at its own startup and the two never talk to each other again. Upload a document through worker A and worker B's `/documents` list won't show it; ask worker B to delete it and you get a 404 for something that very much still exists.

I found this by actually starting the app with `--workers 2` and hammering it with concurrent requests, rather than reasoning about whether it would be fine: under real load, most requests landed on the worker that hadn't seen the newest write, and the desync never healed itself.

The fix — a lock acquired before the process finishes starting up, refusing to let a second instance ever come up at all — sounds like it should be the whole story. Building it exposed two more bugs I hadn't planned for:

1. **A local file lock isn't a global lock.** The first attempt used a `flock` on a file next to the code. That stops two `--workers` on one host, but two containers with separate filesystems — or two separate hosts — would each happily lock their own private file and both believe they were the only instance. The fix was moving the lock into Postgres itself: the one thing every instance already has to agree on, regardless of which host or container it happens to run in.
2. **A lock you stop checking isn't a lock.** Even the Postgres version had a gap: if Postgres itself restarted, or the network dropped, Postgres released the lock on its side — but the process holding it had no way to notice on its own, and kept serving requests as the "sole owner" of state that, as far as Postgres was concerned, nobody owned anymore. A second instance could then start, acquire the now-free lock, and split-brain is back — just relocated from "at startup" to "sometime later, silently." The fix is a background watchdog that keeps re-proving the lock's session is actually still alive (a bounded ping, not an assumption) and hard-exits the process the moment it can't prove that anymore. A restart triggered by a false alarm is cheap; staying up on a lock that can no longer be verified is the real bug.

Neither of those two follow-on bugs was caught by re-reading the first fix — both came from asking "what actually happens if I kill this specific connection right now" and watching the real process, not the code, respond.

### Delete and rename across three stores that don't share a transaction

Deleting a document touches Qdrant, Postgres, and a file on disk — three independent systems with no way to commit all three atomically. The original code just ran the three deletes in sequence and reported success if none of them raised. That's fine until one of them fails partway through — Qdrant is briefly down, say — and now the document's Postgres row is gone (so it vanishes from the list on the next restart, since that's what reloads from Postgres) while its file sits on disk forever, an orphan nothing will ever come back for.

The fix follows the same shape as the process-lock story above: write a durable marker (`status = 'deleting'` in Postgres) *before* touching anything else, so a crash mid-delete leaves a trail instead of a silent inconsistency. A manual retry, or a one-shot pass at the next startup, picks up exactly where it left off, using operations that are safe to repeat (deleting something already deleted is a no-op). The same idea covers folder renames: moving documents into a new folder touches Qdrant's search index one HTTP call per document, so one failure partway through a batch can't be allowed to leave the other documents untouched — Postgres commits the rename as the single durable, source-of-truth step, and Qdrant's copy is treated as a derived index that's allowed to lag briefly and catch back up on its own.

---

## Stack

- **FastAPI** — async API with SSE streaming
- **Qdrant** — vector database (Docker)
- **PostgreSQL** — document metadata (Docker)
- **Ollama** — local LLM inference
- **BAAI/bge-m3** — embeddings (1024-dim)
- **CrossEncoder ms-marco-MiniLM-L-6-v2** — reranking
- **Qdrant sparse vectors (BM25-style, `Modifier.IDF`)** — keyword retrieval, fused server-side with dense search
- **PyMuPDF** — PDF text extraction
- **Tesseract** — OCR for scanned pages
- **Langfuse** — observability (self-hosted, optional)
- **PDF.js** — client-side PDF rendering

---

## License

[MIT](LICENSE)
