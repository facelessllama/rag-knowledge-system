#!/usr/bin/env python3
"""Retrieval-only (no generation) title-free robustness check for
fact_figure_caption — the post-fix 91.5% Evidence-chunk Recall was
measured on questions that always name the paper's own title
(`f'What does {kind} {num} show in the paper "{title}"?'`), which likely
inflates stage-1 ranking (see eval/mixed_corpus/README.md's "Full
taxonomy" caveat). This measures the SAME 177 cases under three question
variants, retrieval-only, against the frozen mixed_corpus_v2 collection:

1. current_golden  — title included, unscoped (what was actually measured
   so far — reproduced here fresh as the same-run baseline for a fair
   three-way comparison, not read from a prior saved file).
2. title_free_global — title dropped, unscoped. An honestly ambiguous
   question over a 762-document corpus (which paper's "Figure 7"?) — a
   stress test, not a fair product metric, since a good system might
   reasonably prefer to ask for clarification over guessing.
3. title_free_scoped — title dropped, retrieve_expanded(document_ids=
   [expected_doc_id]) — the realistic "user already has a document open"
   product workflow. retrieve_expanded currently skips the structural
   Figure/Table-N lookup entirely whenever document_ids is set (`if
   query_structural_refs and not doc_scope`) — this is the number that
   tells us whether that's a real gap.

Usage:
    python eval/mixed_corpus/title_free_retrieval_check.py \\
        --api-url http://localhost:8003 --qdrant-collection mixed_corpus_v2 \\
        --output eval/mixed_corpus/title_free_retrieval_check.json
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
from rag.retriever import (
    HybridRetriever, promote_identity_matches, promote_document_opening_chunks,
    promote_missing_compare_documents,
)
from rag.reranker import CrossEncoderReranker
from rag.query_expander import QueryExpander
from run_eval import evidence_chunk_overlaps

CORPUS_DIR = Path(__file__).resolve().parent
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "3.0"))

VARIANTS = ["current_golden", "title_free_global", "title_free_scoped"]


def title_free_question(case: dict) -> str:
    kind, num = case["caption_label"].split(" ", 1)
    return f"What does {kind} {num} show?"


async def run_variant(variant, case, expected_doc_id, retriever, reranker, query_expander):
    if variant == "current_golden":
        question, document_ids = case["question"], None
    elif variant == "title_free_global":
        question, document_ids = title_free_question(case), None
    else:  # title_free_scoped
        question, document_ids = title_free_question(case), [expected_doc_id]

    expanded = await query_expander.expand(question)
    chunks = await retriever.retrieve_expanded(expanded, top_k=25, document_ids=document_ids)
    if not chunks:
        return {"error": "no chunks retrieved"}

    top_chunks = await asyncio.to_thread(reranker.rerank, question, chunks, 5)
    top_chunks = promote_identity_matches(chunks, top_chunks, RELEVANCE_THRESHOLD)
    top_chunks = promote_document_opening_chunks(chunks, top_chunks)
    top_chunks = promote_missing_compare_documents(chunks, top_chunks, document_ids)

    doc_hit = expected_doc_id in {c.get("document_id") for c in top_chunks}
    hit_pages = [c.get("page_num") for c in top_chunks if c.get("document_id") == expected_doc_id]
    page_hit = case["source_page"] in hit_pages
    same_page_chunks = [
        c for c in top_chunks
        if c.get("document_id") == expected_doc_id and c.get("page_num") == case["source_page"]
        and c.get("char_start") is not None and c.get("char_end") is not None
    ]
    chunk_hit = any(
        evidence_chunk_overlaps(c["char_start"], c["char_end"], case["evidence_char_start"], case["evidence_char_end"])
        for c in same_page_chunks
    )
    structural_present = any(
        c.get("document_id") == expected_doc_id and c.get("source") == "structural_reference"
        for c in top_chunks
    )
    return {
        "doc_hit": doc_hit, "page_hit": page_hit, "chunk_hit": chunk_hit,
        "structural_present": structural_present, "n_sources": len({c.get("document_id") for c in top_chunks}),
    }


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(CORPUS_DIR / "golden_dataset.json"))
    ap.add_argument("--api-url", default=os.getenv("EVAL_API_URL", "http://localhost:8003"))
    ap.add_argument("--api-key", default=os.getenv("API_KEY", ""))
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--qdrant-collection", required=True)
    ap.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    ap.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11435"))
    ap.add_argument("--expander-model", default=os.getenv("QUERY_EXPANDER_MODEL", "qwen2.5:7b"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    assert args.qdrant_collection != "knowledge_base", (
        "Refusing to run against 'knowledge_base' — pass --qdrant-collection explicitly (e.g. mixed_corpus_v2)."
    )

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{args.api_url}/documents", headers=headers, timeout=30.0)
        resp.raise_for_status()
        doc_id_by_filename = {d["filename"]: d["doc_id"] for d in resp.json()["documents"]}

    golden = json.loads(Path(args.dataset).read_text())
    cases = [c for c in golden if c["type"] == "fact_figure_caption"]
    if args.limit:
        cases = cases[: args.limit]

    embedder = EmbeddingService(model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    vector_store = VectorStore(url=args.qdrant_url, collection=args.qdrant_collection, api_key=args.qdrant_api_key)
    retriever = HybridRetriever(embedder, vector_store)
    reranker = CrossEncoderReranker()
    query_expander = QueryExpander(ollama_url=args.ollama_url, model=args.expander_model)

    rows = []
    for i, case in enumerate(cases):
        expected_doc_id = doc_id_by_filename.get(case["expected_doc_filename"])
        row = {"id": case["id"], "expected_doc_id": expected_doc_id}
        if expected_doc_id is None:
            row["error"] = "expected document not found in live /documents"
            rows.append(row)
            continue
        for variant in VARIANTS:
            row[variant] = await run_variant(variant, case, expected_doc_id, retriever, reranker, query_expander)
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(cases)} done", flush=True)

    summary = {}
    for variant in VARIANTS:
        results = [r[variant] for r in rows if variant in r and "error" not in r[variant]]
        n = len(results)
        summary[variant] = {
            "n": n,
            "doc_recall": sum(1 for r in results if r["doc_hit"]) / n if n else None,
            "evidence_page_recall": sum(1 for r in results if r["page_hit"]) / n if n else None,
            "evidence_chunk_recall": sum(1 for r in results if r["chunk_hit"]) / n if n else None,
            "structural_present_rate": sum(1 for r in results if r["structural_present"]) / n if n else None,
        }

    output = {"summary": summary, "rows": rows}
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'Variant':<20} {'n':>4} {'DocRecall':>10} {'EvPageRec':>10} {'EvChunkRec':>11} {'StructPresent':>14}")
    for variant in VARIANTS:
        s = summary[variant]
        print(f"{variant:<20} {s['n']:>4} {s['doc_recall']*100:>9.1f}% {s['evidence_page_recall']*100:>9.1f}% "
              f"{s['evidence_chunk_recall']*100:>10.1f}% {s['structural_present_rate']*100:>13.1f}%")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
