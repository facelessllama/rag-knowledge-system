#!/usr/bin/env python3
"""Captures frozen retrieval contexts for the title-free, document-scoped
generation check — deliberately a SEPARATE artifact from the main post-fix
A/B (eval/mixed_corpus/generator_ab_postfix_v3_contexts.json), never
merged into it: dropping the paper title changes the prompt itself, so
mixing the two populations would conflate a retrieval-scope difference
with a wording difference in one number.

Same idea as capture_generation_contexts.py (retrieval run once per case
against the frozen mixed_corpus_v2 collection, evidence_chunk_overlaps()
re-verified before keeping it), with two differences matching the
realistic "user already has a document open" product workflow this
checks (see eval/mixed_corpus/title_free_retrieval_check.py, whose
question wording this reuses):
1. The question drops the paper title (`title_free_question()`).
2. retrieve_expanded() is called with document_ids=[expected_doc_id] —
   the realistic scope a real product request would carry.

Usage:
    python eval/mixed_corpus/capture_title_free_scoped_contexts.py \\
        --dataset eval/mixed_corpus/golden_dataset_v3.json \\
        --qdrant-collection mixed_corpus_v2 \\
        --output eval/mixed_corpus/generator_ab_postfix_v3_titlefreescoped_contexts.json
"""
import argparse
import asyncio
import json
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

import httpx

from embeddings.embedding_service import EmbeddingService
from vector_db.qdrant_client import VectorStore
from rag.retriever import HybridRetriever, promote_identity_matches, promote_document_opening_chunks
from rag.reranker import CrossEncoderReranker
from rag.prompt_builder import PromptBuilder
from rag.query_expander import QueryExpander
from run_eval import evidence_chunk_overlaps
from title_free_retrieval_check import title_free_question

CORPUS_DIR = Path(__file__).resolve().parent
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "3.0"))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(CORPUS_DIR / "golden_dataset_v3.json"))
    ap.add_argument("--api-url", default=os.getenv("EVAL_API_URL", "http://localhost:8003"))
    ap.add_argument("--api-key", default=os.getenv("API_KEY", ""))
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--qdrant-collection", required=True)
    ap.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    ap.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11435"))
    ap.add_argument("--expander-model", default=os.getenv("QUERY_EXPANDER_MODEL", "qwen2.5:7b"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    assert args.qdrant_collection != "knowledge_base", (
        "Refusing to run against 'knowledge_base' (the main production collection) — "
        "pass --qdrant-collection explicitly (e.g. mixed_corpus_v2)."
    )

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{args.api_url}/documents", headers=headers, timeout=30.0)
        resp.raise_for_status()
        doc_id_by_filename = {d["filename"]: d["doc_id"] for d in resp.json()["documents"]}

    golden = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    cases = [c for c in golden if c["type"] == "fact_figure_caption"]
    if args.limit:
        cases = cases[: args.limit]

    embedder = EmbeddingService(model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    vector_store = VectorStore(url=args.qdrant_url, collection=args.qdrant_collection, api_key=args.qdrant_api_key)
    retriever = HybridRetriever(embedder, vector_store)
    reranker = CrossEncoderReranker()
    prompt_builder = PromptBuilder()
    query_expander = QueryExpander(ollama_url=args.ollama_url, model=args.expander_model)

    coll_info = vector_store.get_collection_info()

    contexts = []
    excluded = []
    for i, case in enumerate(cases):
        expected_doc_id = doc_id_by_filename.get(case["expected_doc_filename"])
        if expected_doc_id is None:
            excluded.append({"id": case["id"], "reason": "expected document not found in live /documents"})
            continue

        q = title_free_question(case)
        expanded = await query_expander.expand(q)
        chunks = await retriever.retrieve_expanded(expanded, top_k=args.top_k * 5, document_ids=[expected_doc_id])
        if not chunks:
            excluded.append({"id": case["id"], "reason": "no chunks retrieved"})
            continue

        top_chunks = await asyncio.to_thread(reranker.rerank, q, chunks, args.top_k)
        top_chunks = promote_identity_matches(chunks, top_chunks, RELEVANCE_THRESHOLD)
        top_chunks = promote_document_opening_chunks(chunks, top_chunks)

        ev_start, ev_end = case.get("evidence_char_start"), case.get("evidence_char_end")
        still_hit = any(
            c.get("document_id") == expected_doc_id and c.get("page_num") == case.get("source_page")
            and c.get("char_start") is not None and c.get("char_end") is not None
            and evidence_chunk_overlaps(c["char_start"], c["char_end"], ev_start, ev_end)
            for c in top_chunks
        )
        if not still_hit:
            excluded.append({"id": case["id"], "reason": "no confirmed evidence_chunk_hit under title-free scoped retrieval"})
            print(f"[{i+1}/{len(cases)}] {case['id']}: EXCLUDED (no hit)", flush=True)
            continue

        messages = prompt_builder.build(query=q, chunks=top_chunks)
        contexts.append({
            "id": case["id"],
            "type": case["type"],
            "question": q,
            "expected_substring": case.get("expected_substring"),
            "messages": messages,
            "meta": {
                "expected_doc_filename": case.get("expected_doc_filename"),
                "source_page": case.get("source_page"),
                "n_chunks": len(top_chunks),
                "label_status": case.get("label_status"),
                "document_scoped": True,
                "title_free": True,
            },
        })
        print(f"[{i+1}/{len(cases)}] {case['id']}: captured ({len(top_chunks)} chunks)", flush=True)

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
            "qdrant_collection": args.qdrant_collection,
            "qdrant_points_count": coll_info.get("total_vectors"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "query_expander_model": args.expander_model,
            "top_k": args.top_k,
            "relevance_threshold": RELEVANCE_THRESHOLD,
            "scenario": "title_free_scoped",
            "n_captured": len(contexts),
            "n_excluded": len(excluded),
        },
        "excluded": excluded,
        "contexts": contexts,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nCaptured {len(contexts)}/{len(cases)} contexts ({len(excluded)} excluded)")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
