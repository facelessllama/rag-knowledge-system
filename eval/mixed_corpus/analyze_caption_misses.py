#!/usr/bin/env python3
"""Full automated taxonomy of the remaining `fact_figure_caption` Evidence-
chunk Recall misses (see project memory: structural lookup raised this to
28.2%, but ~72% of caption questions still miss). Per the user's explicit
instruction, this is a diagnostic pass to find out WHICH failure mode
dominates before picking a next retrieval mechanism — not another blind
mechanism attempt (see the reverted two-stage pool-widening attempt and
the STRUCTURAL_TOP_DOCS 3->5 widening, both tried on a single deeply-
investigated case and both failed to generalize).

Re-runs the real production sequence once per miss case, in-process,
exactly like capture_generation_contexts.py already does (same
nondeterminism safeguard: a case that no longer reproduces as a miss on
this fresh run is excluded, not silently counted). Two extra things this
script needs that capture_generation_contexts.py doesn't:

1. Which documents retrieve_expanded()'s structural lookup actually
   text-scanned (its STRUCTURAL_TOP_DOCS=3 window) and what, if anything,
   best_structural_chunk() found in each — captured by wrapping (not
   modifying) VectorStore.chunks_for_document and rag.retriever.
   best_structural_chunk with recording wrappers around the real calls,
   restored after every case.
2. A corpus-wide caption-label collision index (build_caption_label_
   index.py) — whether OTHER documents in the corpus also contain a
   caption-tier match for the same Figure/Table number, so "collisions
   displaced the correct document" is a measured fact instead of a guess
   from one hand-investigated case.

Usage:
    python eval/mixed_corpus/analyze_caption_misses.py \\
        --scores eval/mixed_corpus/last_run_scores_golden_dataset_v2_structural.json \\
        --caption-index eval/mixed_corpus/caption_label_index.json \\
        --output eval/mixed_corpus/caption_miss_taxonomy.json
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from embeddings.embedding_service import EmbeddingService
from vector_db.qdrant_client import VectorStore
import rag.retriever as retriever_module
from rag.retriever import (
    HybridRetriever, promote_identity_matches, promote_document_opening_chunks,
    promote_missing_compare_documents,
)
from rag.reranker import CrossEncoderReranker
from rag.query_expander import QueryExpander
from run_eval import evidence_chunk_overlaps

CORPUS_DIR = Path(__file__).resolve().parent
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "3.0"))

# Fixed taxonomy — designed before coding, per the user's explicit
# instruction not to let a mechanism idea drive which categories exist.
# Every miss lands in exactly one of these (or "uncategorized", reported
# separately rather than forced into a bad fit).
CATEGORY_MECHANISM = {
    "correct_doc_below_candidate_window": "document disambiguation",
    "multiple_docs_share_figure_n": "collision-aware scoring",
    "caption_absent_from_indexed_chunks": "parser/chunker",
    "prose_or_wrong_occurrence_picked": "caption-aware scoring",
    "structural_chunk_found_but_displaced": "context assembly",
    "uncategorized": "needs manual review",
}


def categorize(row: dict) -> str:
    if not row["in_structural_window"]:
        return "multiple_docs_share_figure_n" if row["collision_in_scanned_window"] else "correct_doc_below_candidate_window"
    if not row["caption_indexed_in_doc"]:
        return "caption_absent_from_indexed_chunks"
    if not row["structural_match_found"]:
        # caption_indexed_in_doc (offline scan) and structural_match_found
        # (live best_structural_chunk call) use the same detection function
        # against the same chunk set — disagreement here means the corpus
        # changed between the two scans, not a real taxonomy category.
        return "uncategorized"
    if not row["structural_match_correct_span"]:
        return "prose_or_wrong_occurrence_picked"
    if not row["structural_chunk_in_final_top"]:
        return "structural_chunk_found_but_displaced"
    return "uncategorized"


async def analyze_case(case, doc_id_by_filename, caption_index, vector_store, retriever, reranker, query_expander):
    expected_fname = case["expected_doc_filename"]
    expected_doc_id = doc_id_by_filename.get(expected_fname)
    if expected_doc_id is None:
        return None, "expected document not found in live /documents (not ingested?)"

    kind, num = case["caption_label"].split(" ", 1)
    source_page = case["source_page"]
    ev_start, ev_end = case["evidence_char_start"], case["evidence_char_end"]

    scanned_doc_ids: list[str] = []
    match_by_doc: dict[str, dict | None] = {}
    seen_in_hybrid_pool: dict[str, float] = {}

    original_chunks_for_document = vector_store.chunks_for_document
    original_best_structural_chunk = retriever_module.best_structural_chunk
    original_hybrid_search = vector_store.hybrid_search

    def recording_chunks_for_document(document_id, limit=2000):
        scanned_doc_ids.append(document_id)
        return original_chunks_for_document(document_id, limit)

    def recording_best_structural_chunk(chunks, k, n):
        doc_id = chunks[0].get("document_id") if chunks else None
        result = original_best_structural_chunk(chunks, k, n)
        match_by_doc[doc_id] = result
        return result

    def recording_hybrid_search(*a, **kw):
        # retrieve_expanded's identity-lookup blocks (case/CELEX/citation/
        # structural) all share one guard: "if key not in all_chunks: inject
        # with score=1.0/identity_match=True" — silently a no-op if the SAME
        # chunk was already found (however weakly) by plain hybrid search.
        # case/CELEX/citation numbers have a second, independent path that
        # still marks identity_match=True from the chunk's OWN stored
        # metadata regardless of this guard; a caption has no such stored
        # metadata (see rag/retriever.py's own comment on this), so this is
        # the only path a caption chunk has — recording every chunk_id/score
        # plain hybrid search ever surfaces lets us tell whether THIS
        # specific guard-skip is what displaced a given case.
        results = original_hybrid_search(*a, **kw)
        for r in results:
            cid = r.get("chunk_id")
            if cid and (cid not in seen_in_hybrid_pool or r["score"] > seen_in_hybrid_pool[cid]):
                seen_in_hybrid_pool[cid] = r["score"]
        return results

    with patch.object(vector_store, "chunks_for_document", side_effect=recording_chunks_for_document), \
         patch.object(retriever_module, "best_structural_chunk", side_effect=recording_best_structural_chunk), \
         patch.object(vector_store, "hybrid_search", side_effect=recording_hybrid_search):
        expanded = await query_expander.expand(case["question"])
        chunks = await retriever.retrieve_expanded(expanded, top_k=25)

    if not chunks:
        return None, "no chunks retrieved on this fresh run"

    top_chunks = await asyncio.to_thread(reranker.rerank, case["question"], chunks, 5)
    top_chunks = promote_identity_matches(chunks, top_chunks, RELEVANCE_THRESHOLD)
    top_chunks = promote_document_opening_chunks(chunks, top_chunks)
    top_chunks = promote_missing_compare_documents(chunks, top_chunks, None)

    # Recompute the same miss condition run_eval.py scored, on THIS fresh
    # run — a case whose fresh result no longer reproduces as a miss is
    # cross-process retrieval nondeterminism (see project memory), not a
    # real taxonomy row, and must be excluded rather than silently kept.
    hit_pages = [c.get("page_num") for c in top_chunks if c.get("document_id") == expected_doc_id]
    evidence_page_hit_fresh = source_page in hit_pages
    same_page_chunks = [
        c for c in top_chunks
        if c.get("document_id") == expected_doc_id and c.get("page_num") == source_page
        and c.get("char_start") is not None and c.get("char_end") is not None
    ]
    evidence_chunk_hit_fresh = any(
        evidence_chunk_overlaps(c["char_start"], c["char_end"], ev_start, ev_end) for c in same_page_chunks
    )
    if evidence_chunk_hit_fresh:
        return None, "no longer a miss on this fresh run (nondeterminism)"

    doc_ids_in_chunks = {c.get("document_id") for c in chunks}
    doc_present_in_chunks = expected_doc_id in doc_ids_in_chunks
    by_score = sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)
    doc_order = []
    for c in by_score:
        d = c.get("document_id")
        if d not in doc_order:
            doc_order.append(d)
    doc_rank_in_chunks = doc_order.index(expected_doc_id) + 1 if expected_doc_id in doc_order else None

    label_key = [kind, num]
    caption_indexed_in_doc = label_key in caption_index.get(expected_doc_id, [])
    collision_count = sum(
        1 for doc_id, labels in caption_index.items()
        if doc_id != expected_doc_id and label_key in labels
    )
    collision_in_scanned_window = any(
        d != expected_doc_id and label_key in caption_index.get(d, [])
        for d in scanned_doc_ids
    )

    in_structural_window = expected_doc_id in scanned_doc_ids
    structural_match = match_by_doc.get(expected_doc_id)
    structural_match_found = structural_match is not None
    structural_match_correct_span = None
    if structural_match_found:
        structural_match_correct_span = (
            structural_match.get("page_num") == source_page
            and structural_match.get("char_start") is not None
            and structural_match.get("char_end") is not None
            and evidence_chunk_overlaps(structural_match["char_start"], structural_match["char_end"], ev_start, ev_end)
        )
    structural_chunk_in_final_top = any(
        c.get("document_id") == expected_doc_id and c.get("source") == "structural_reference"
        for c in top_chunks
    )
    # Only meaningful (and only checked) for the "found but displaced"
    # bucket — see recording_hybrid_search's docstring above.
    winning_chunk_already_in_hybrid_pool = None
    if structural_match_found and structural_match_correct_span and not structural_chunk_in_final_top:
        winning_chunk_id = structural_match.get("chunk_id")
        winning_chunk_already_in_hybrid_pool = winning_chunk_id in seen_in_hybrid_pool

    row = {
        "id": case["id"],
        "caption_label": case["caption_label"],
        "expected_doc_filename": expected_fname,
        "doc_present_in_chunks": doc_present_in_chunks,
        "doc_rank_in_chunks": doc_rank_in_chunks,
        "in_structural_window": in_structural_window,
        "scanned_doc_ids": scanned_doc_ids,
        "collision_count": collision_count,
        "collision_in_scanned_window": collision_in_scanned_window,
        "caption_indexed_in_doc": caption_indexed_in_doc,
        "structural_match_found": structural_match_found,
        "structural_match_correct_span": structural_match_correct_span,
        "structural_chunk_in_final_top": structural_chunk_in_final_top,
        "winning_chunk_already_in_hybrid_pool": winning_chunk_already_in_hybrid_pool,
        "evidence_page_hit_fresh": evidence_page_hit_fresh,
    }
    row["category"] = categorize(row)
    return row, None


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(CORPUS_DIR / "golden_dataset.json"))
    ap.add_argument("--scores", required=True,
                     help="A run_eval.py --output scores file to select miss case IDs from "
                          "(fact_figure_caption cases where evidence_chunk_hit is not truthy)")
    ap.add_argument("--caption-index", default=str(CORPUS_DIR / "caption_label_index.json"),
                     help="Output of build_caption_label_index.py")
    ap.add_argument("--api-url", default=os.getenv("EVAL_API_URL", "http://localhost:8003"))
    ap.add_argument("--api-key", default=os.getenv("API_KEY", ""))
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--qdrant-collection", required=True,
                     help="No default on purpose — same guard as capture_generation_contexts.py.")
    ap.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    ap.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11435"))
    ap.add_argument("--expander-model", default=os.getenv("QUERY_EXPANDER_MODEL", "qwen2.5:7b"))
    ap.add_argument("--limit", type=int, default=None, help="Cap the number of cases (for a pilot run)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    assert args.qdrant_collection != "knowledge_base", (
        "Refusing to run against 'knowledge_base' (the main production collection) — "
        "pass --qdrant-collection explicitly (e.g. mixed_corpus_v2)."
    )

    import httpx
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{args.api_url}/documents", headers=headers, timeout=30.0)
        resp.raise_for_status()
        doc_id_by_filename = {d["filename"]: d["doc_id"] for d in resp.json()["documents"]}

    golden = {c["id"]: c for c in json.loads(Path(args.dataset).read_text())}
    scores = json.loads(Path(args.scores).read_text())
    miss_ids = [
        d["id"] for d in scores
        if d["type"] == "fact_figure_caption" and not d.get("evidence_chunk_hit")
    ]
    if args.limit:
        miss_ids = miss_ids[: args.limit]
    print(f"{len(miss_ids)} candidate miss cases", flush=True)

    caption_index_raw = json.loads(Path(args.caption_index).read_text())
    caption_index = {doc_id: [list(pair) for pair in pairs] for doc_id, pairs in caption_index_raw.items()}

    embedder = EmbeddingService(model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    vector_store = VectorStore(url=args.qdrant_url, collection=args.qdrant_collection, api_key=args.qdrant_api_key)
    retriever = HybridRetriever(embedder, vector_store)
    reranker = CrossEncoderReranker()
    query_expander = QueryExpander(ollama_url=args.ollama_url, model=args.expander_model)

    rows = []
    excluded = []
    for i, cid in enumerate(miss_ids):
        case = golden[cid]
        row, reason = await analyze_case(case, doc_id_by_filename, caption_index, vector_store, retriever, reranker, query_expander)
        if row is None:
            excluded.append({"id": cid, "reason": reason})
            print(f"[{i+1}/{len(miss_ids)}] {cid}: EXCLUDED ({reason})", flush=True)
        else:
            rows.append(row)
            print(f"[{i+1}/{len(miss_ids)}] {cid}: {row['category']}", flush=True)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1

    output = {
        "run_metadata": {
            "scores_source": args.scores,
            "caption_index_source": args.caption_index,
            "qdrant_collection": args.qdrant_collection,
            "n_candidate_misses": len(miss_ids),
            "n_analyzed": len(rows),
            "n_excluded": len(excluded),
        },
        "category_counts": counts,
        "excluded": excluded,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(rows)}/{len(miss_ids)} analyzed ({len(excluded)} excluded)\n")
    print(f"{'Category':<38} {'n':>4}  Next mechanism")
    for cat, mechanism in CATEGORY_MECHANISM.items():
        n = counts.get(cat, 0)
        print(f"{cat:<38} {n:>4}  {mechanism}")

    displaced = [r for r in rows if r["category"] == "structural_chunk_found_but_displaced"]
    if displaced:
        guard_skip = sum(1 for r in displaced if r["winning_chunk_already_in_hybrid_pool"])
        print(f"\nOf {len(displaced)} 'structural_chunk_found_but_displaced' cases, "
              f"{guard_skip} had the winning chunk already present (at a weak score) in the "
              f"raw hybrid pool — the identity-boost guard in retrieve_expanded's structural "
              f"lookup ('if key not in all_chunks') silently no-ops in exactly this situation, "
              f"unlike case/CELEX/citation-number matches which have a second, metadata-based "
              f"path that still marks identity_match=True regardless of this guard.")

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
