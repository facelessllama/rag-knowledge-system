#!/usr/bin/env python3
"""Captures frozen retrieval contexts (the exact `messages` prompt a
generator would receive) for a set of golden-dataset cases, so a
generator A/B (see eval/compare_generators.py) can run multiple models
against IDENTICAL input without ever touching Qdrant/embeddings again —
retrieval and generation are measured separately on purpose.

Runs the real production sequence once per case: query_expander.expand ->
retriever.retrieve_expanded -> reranker.rerank -> promote_identity_matches
-> prompt_builder.build. Safeguard: re-verifies evidence_chunk_overlaps()
against the freshly-built context before capturing it — a cross-process
retrieval nondeterminism was found and confirmed this session (see
project memory: a close top-3/top-4 stage-1 ranking boundary can flip
between process instances even on a frozen collection), so a case whose
fresh context no longer contains the expected evidence is EXCLUDED and
reported, never silently captured as if it were still "confirmed."

Usage:
    python eval/mixed_corpus/capture_generation_contexts.py \\
        --scores eval/mixed_corpus/last_run_scores_golden_dataset_v2_structural.json \\
        --type fact_figure_caption --require-field evidence_chunk_hit \\
        --output /tmp/contexts.json
"""
import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from dotenv import load_dotenv
load_dotenv()

from embeddings.embedding_service import EmbeddingService
from vector_db.qdrant_client import VectorStore
from rag.retriever import HybridRetriever, promote_identity_matches
from rag.reranker import CrossEncoderReranker
from rag.prompt_builder import PromptBuilder
from rag.query_expander import QueryExpander
from run_eval import evidence_chunk_overlaps

CORPUS_DIR = Path(__file__).resolve().parent
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "3.0"))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(CORPUS_DIR / "golden_dataset.json"))
    ap.add_argument("--scores", required=True,
                     help="A run_eval.py --output scores file to select case IDs from")
    ap.add_argument("--type", default="fact_figure_caption")
    ap.add_argument("--require-field", default="evidence_chunk_hit",
                     help="Only capture cases where this field is truthy in --scores")
    ap.add_argument("--baseline-scores", default=None,
                     help="Optional second scores file (e.g. a pre-retrieval-change baseline) — cases "
                          "where --require-field is truthy in --scores but NOT in this file get "
                          "meta.saved_by_change=true, so compare_generators.py --breakdown-by can "
                          "report that subset separately (e.g. 'cases the structural lookup rescued')")
    ap.add_argument("--limit", type=int, default=None, help="Cap the number of cases (for a pilot run)")
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--qdrant-collection", default=os.getenv("QDRANT_COLLECTION", "mixed_corpus_v2"))
    ap.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    ap.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11435"))
    ap.add_argument("--expander-model", default=os.getenv("QUERY_EXPANDER_MODEL", "qwen2.5:7b"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    golden = {c["id"]: c for c in json.loads(Path(args.dataset).read_text())}
    scores = json.loads(Path(args.scores).read_text())
    case_ids = [
        d["id"] for d in scores
        if d["type"] == args.type and d.get(args.require_field)
    ]
    if args.limit:
        case_ids = case_ids[: args.limit]
    print(f"{len(case_ids)} candidate cases (type={args.type}, {args.require_field} truthy)", flush=True)

    saved_by_change = set()
    if args.baseline_scores:
        baseline = {d["id"]: d for d in json.loads(Path(args.baseline_scores).read_text())}
        saved_by_change = {
            cid for cid in case_ids
            if not baseline.get(cid, {}).get(args.require_field)
        }
        print(f"{len(saved_by_change)} of these were NOT truthy in --baseline-scores (saved_by_change)", flush=True)

    # --qdrant-collection's own default reads QDRANT_COLLECTION from the
    # environment (via load_dotenv() above) — .env sets that to
    # "knowledge_base" (the main production collection) for the API
    # server's own use, so an invocation that forgets --qdrant-collection
    # explicitly would silently query the wrong corpus entirely rather
    # than erroring (confirmed: cost a debugging session — every capture
    # came back "excluded" because the main collection has none of these
    # documents at all, not because of retrieval nondeterminism).
    assert args.qdrant_collection != "knowledge_base", (
        "Refusing to run against 'knowledge_base' (the main production collection) — "
        "pass --qdrant-collection explicitly (e.g. mixed_corpus_v2)."
    )
    embedder = EmbeddingService(model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    vector_store = VectorStore(url=args.qdrant_url, collection=args.qdrant_collection, api_key=args.qdrant_api_key)
    retriever = HybridRetriever(embedder, vector_store)
    reranker = CrossEncoderReranker()
    prompt_builder = PromptBuilder()
    query_expander = QueryExpander(ollama_url=args.ollama_url, model=args.expander_model)

    coll_info = vector_store.get_collection_info()

    contexts = []
    excluded = []
    for i, cid in enumerate(case_ids):
        g = golden[cid]
        q = g["question"]

        expanded = await query_expander.expand(q)
        chunks = await retriever.retrieve_expanded(expanded, top_k=25)
        if not chunks:
            excluded.append({"id": cid, "reason": "no chunks retrieved"})
            continue

        top_chunks = await asyncio.to_thread(reranker.rerank, q, chunks, args.top_k)
        top_chunks = promote_identity_matches(chunks, top_chunks, RELEVANCE_THRESHOLD)

        ev_start, ev_end = g.get("evidence_char_start"), g.get("evidence_char_end")
        doc_id = next((c["document_id"] for c in top_chunks if c.get("filename") == g["expected_doc_filename"]), None)
        still_hit = doc_id is not None and any(
            c.get("document_id") == doc_id and c.get("page_num") == g.get("source_page")
            and c.get("char_start") is not None and c.get("char_end") is not None
            and evidence_chunk_overlaps(c["char_start"], c["char_end"], ev_start, ev_end)
            for c in top_chunks
        )
        if not still_hit:
            excluded.append({"id": cid, "reason": "fresh context no longer contains the evidence chunk (nondeterminism)"})
            print(f"[{i+1}/{len(case_ids)}] {cid}: EXCLUDED (context drift)", flush=True)
            continue

        messages = prompt_builder.build(query=q, chunks=top_chunks)
        contexts.append({
            "id": cid,
            "type": g["type"],
            "question": q,
            "expected_substring": g.get("expected_substring"),
            "messages": messages,
            "meta": {
                "expected_doc_filename": g.get("expected_doc_filename"),
                "source_page": g.get("source_page"),
                "n_chunks": len(top_chunks),
                "saved_by_change": cid in saved_by_change,
            },
        })
        print(f"[{i+1}/{len(case_ids)}] {cid}: captured ({len(top_chunks)} chunks)", flush=True)

    manifest_path = CORPUS_DIR / "manifest.json"
    dataset_path = Path(args.dataset)
    import hashlib
    output = {
        "run_metadata": {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "dataset_path": str(dataset_path),
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.exists() else None,
            "scores_source": args.scores,
            "qdrant_collection": args.qdrant_collection,
            "qdrant_points_count": coll_info.get("total_vectors"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "query_expander_model": args.expander_model,
            "top_k": args.top_k,
            "relevance_threshold": RELEVANCE_THRESHOLD,
            "n_captured": len(contexts),
            "n_excluded": len(excluded),
        },
        "excluded": excluded,
        "contexts": contexts,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nCaptured {len(contexts)}/{len(case_ids)} contexts ({len(excluded)} excluded)")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
