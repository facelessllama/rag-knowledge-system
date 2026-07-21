#!/usr/bin/env python3
"""Controlled factor decomposition for the title-bearing-vs-title-free-
scoped paradox (post-fix DeepSeek A/B: DeepSeek's ~21-26pp edge shrinks
to ~+4pp/~0pp without the title — see project memory / EXPERIMENT_
HISTORY.md). NOT a new full A/B run — no generator model is called here.
Only local retrieval is re-run (embedding + Qdrant + rerank — free,
deterministic-enough-for-this-purpose, the same "still_hit" re-
verification safeguard as every other script in this family), to recover
per-chunk data (evidence chunk identity, its rank, source count, context
length) that the two capture scripts didn't persist, plus a join of the
ALREADY-SAVED generation results (no new calls) for correct<->wrong
transition counts between the two scenarios.

Usage:
    python eval/mixed_corpus/cross_run_factor_analysis.py \\
        --qdrant-collection mixed_corpus_v2 \\
        --output eval/mixed_corpus/cross_run_factor_analysis_report.json
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from embeddings.embedding_service import EmbeddingService
from vector_db.qdrant_client import VectorStore
from rag.retriever import HybridRetriever, promote_identity_matches, promote_document_opening_chunks
from rag.reranker import CrossEncoderReranker
from rag.query_expander import QueryExpander
from run_eval import evidence_chunk_overlaps
from title_free_retrieval_check import title_free_question

CORPUS_DIR = Path(__file__).resolve().parent
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "3.0"))


async def probe(case, expected_doc_id, question, document_ids, retriever, reranker, query_expander, top_k=5):
    expanded = await query_expander.expand(question)
    chunks = await retriever.retrieve_expanded(expanded, top_k=top_k * 5, document_ids=document_ids)
    if not chunks:
        return None
    top_chunks = await asyncio.to_thread(reranker.rerank, question, chunks, top_k)
    top_chunks = promote_identity_matches(chunks, top_chunks, RELEVANCE_THRESHOLD)
    top_chunks = promote_document_opening_chunks(chunks, top_chunks)

    ev_start, ev_end = case.get("evidence_char_start"), case.get("evidence_char_end")
    source_page = case.get("source_page")
    # rank = position (1-indexed) by score among unique documents in top_chunks, matching
    # this project's existing "doc_rank_in_chunks" convention (analyze_caption_misses.py)
    sorted_chunks = sorted(top_chunks, key=lambda c: c.get("rerank_score", c.get("score", 0)), reverse=True)
    evidence_chunk = None
    evidence_rank = None
    for i, c in enumerate(sorted_chunks):
        if (c.get("document_id") == expected_doc_id and c.get("page_num") == source_page
                and c.get("char_start") is not None and c.get("char_end") is not None
                and evidence_chunk_overlaps(c["char_start"], c["char_end"], ev_start, ev_end)):
            evidence_chunk = c
            evidence_rank = i + 1
            break
    if evidence_chunk is None:
        return {"hit": False}

    context_chars = sum(len(c.get("text", "")) for c in top_chunks)
    return {
        "hit": True,
        "n_sources": len({c.get("document_id") for c in top_chunks}),
        "n_chunks": len(top_chunks),
        "context_chars": context_chars,
        "evidence_chunk_id": evidence_chunk.get("chunk_id"),
        "evidence_rank": evidence_rank,
        "evidence_source": evidence_chunk.get("source"),
    }


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden-v3", default=str(CORPUS_DIR / "golden_dataset_v3.json"))
    ap.add_argument("--titlebearing-contexts", default=str(CORPUS_DIR / "generator_ab_postfix_v3_contexts.json"))
    ap.add_argument("--titlefree-contexts", default=str(CORPUS_DIR / "generator_ab_postfix_v3_titlefreescoped_contexts.json"))
    ap.add_argument("--titlebearing-results", default=str(CORPUS_DIR / "generator_ab_postfix_v3_results.json"))
    ap.add_argument("--titlefree-results", default=str(CORPUS_DIR / "generator_ab_postfix_v3_titlefreescoped_results.json"))
    ap.add_argument("--api-url", default=os.getenv("EVAL_API_URL", "http://localhost:8003"))
    ap.add_argument("--api-key", default=os.getenv("API_KEY", ""))
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--qdrant-collection", required=True)
    ap.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    ap.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11435"))
    ap.add_argument("--expander-model", default=os.getenv("QUERY_EXPANDER_MODEL", "qwen2.5:7b"))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    assert args.qdrant_collection != "knowledge_base"

    golden_v3 = {c["id"]: c for c in json.loads(Path(args.golden_v3).read_text(encoding="utf-8"))}
    tb_ctx_ids = {c["id"] for c in json.loads(Path(args.titlebearing_contexts).read_text(encoding="utf-8"))["contexts"]}
    tf_ctx_ids = {c["id"] for c in json.loads(Path(args.titlefree_contexts).read_text(encoding="utf-8"))["contexts"]}
    intersection_ids = sorted(tb_ctx_ids & tf_ctx_ids)
    print(f"{len(tb_ctx_ids)} title-bearing evidence-hit, {len(tf_ctx_ids)} title-free-scoped evidence-hit, "
          f"{len(intersection_ids)} in both", flush=True)

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{args.api_url}/documents", headers=headers, timeout=30.0)
        resp.raise_for_status()
        doc_id_by_filename = {d["filename"]: d["doc_id"] for d in resp.json()["documents"]}

    embedder = EmbeddingService(model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    vector_store = VectorStore(url=args.qdrant_url, collection=args.qdrant_collection, api_key=args.qdrant_api_key)
    retriever = HybridRetriever(embedder, vector_store)
    reranker = CrossEncoderReranker()
    query_expander = QueryExpander(ollama_url=args.ollama_url, model=args.expander_model)

    rows = []
    for i, cid in enumerate(intersection_ids):
        case = golden_v3[cid]
        expected_doc_id = doc_id_by_filename.get(case["expected_doc_filename"])
        if expected_doc_id is None:
            continue
        tb = await probe(case, expected_doc_id, case["question"], None, retriever, reranker, query_expander)
        tf = await probe(case, expected_doc_id, title_free_question(case), [expected_doc_id], retriever, reranker, query_expander)
        rows.append({
            "id": cid,
            "label_status": case.get("label_status"),
            "caption_len": len(case.get("expected_substring") or ""),
            "title_bearing": tb,
            "title_free_scoped": tf,
        })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(intersection_ids)} done", flush=True)

    # ── transition analysis from ALREADY-SAVED generation results (no new calls) ──
    tb_results = {(r["id"], r["model_label"]): r for r in json.loads(Path(args.titlebearing_results).read_text(encoding="utf-8"))["results"]}
    tf_results = {(r["id"], r["model_label"]): r for r in json.loads(Path(args.titlefree_results).read_text(encoding="utf-8"))["results"]}
    models = sorted({k[1] for k in tb_results} & {k[1] for k in tf_results})

    transitions = {}
    for model in models:
        tb_to_tf = {"correct_to_correct": 0, "correct_to_wrong": 0, "wrong_to_correct": 0, "wrong_to_wrong": 0}
        for cid in intersection_ids:
            a, b = tb_results.get((cid, model)), tf_results.get((cid, model))
            if a is None or b is None:
                continue
            if a["correct"] and b["correct"]:
                tb_to_tf["correct_to_correct"] += 1
            elif a["correct"] and not b["correct"]:
                tb_to_tf["correct_to_wrong"] += 1
            elif not a["correct"] and b["correct"]:
                tb_to_tf["wrong_to_correct"] += 1
            else:
                tb_to_tf["wrong_to_wrong"] += 1
        transitions[model] = tb_to_tf

    # ── aggregate factor comparison, verified-tier only ──
    verified_rows = [r for r in rows if r["label_status"] in ("verified", "verified_reviewed", "extracted")
                      and r["title_bearing"] and r["title_bearing"]["hit"]
                      and r["title_free_scoped"] and r["title_free_scoped"]["hit"]]

    def avg(key, side):
        vals = [r[side][key] for r in verified_rows if r[side].get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    same_chunk = sum(
        1 for r in verified_rows
        if r["title_bearing"]["evidence_chunk_id"] == r["title_free_scoped"]["evidence_chunk_id"]
    )

    summary = {
        "n_intersection": len(intersection_ids),
        "n_both_hit_verified_tier": len(verified_rows),
        "same_evidence_chunk_id": same_chunk,
        "same_evidence_chunk_pct": round(100 * same_chunk / len(verified_rows), 1) if verified_rows else None,
        "avg_n_sources": {"title_bearing": avg("n_sources", "title_bearing"), "title_free_scoped": avg("n_sources", "title_free_scoped")},
        "avg_context_chars": {"title_bearing": avg("context_chars", "title_bearing"), "title_free_scoped": avg("context_chars", "title_free_scoped")},
        "avg_evidence_rank": {"title_bearing": avg("evidence_rank", "title_bearing"), "title_free_scoped": avg("evidence_rank", "title_free_scoped")},
        "transitions_title_bearing_to_title_free": transitions,
    }

    output = {"summary": summary, "rows": rows}
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{json.dumps(summary, indent=2)}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
