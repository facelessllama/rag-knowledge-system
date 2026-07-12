#!/usr/bin/env python3
"""
Runs eval/golden_dataset.json against a live instance and reports:

- Recall@K (was the expected document among the K sources returned?)
- MRR (mean reciprocal rank of the expected document among sources)
- abstention accuracy (did the system correctly answer vs. correctly refuse?)
- answer correctness (does the answer contain the expected fact substring?)

Also dumps every case's retrieval score and rerank score to
eval/last_run_scores.json — used by eval/calibrate_threshold.py to pick
RELEVANCE_THRESHOLD from real score distributions instead of a guess.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --api-url http://localhost:8000 --top-k 5
"""
import argparse
import asyncio
import json
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv
import os

load_dotenv()

DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"
SCORES_OUT_PATH = Path(__file__).resolve().parent / "last_run_scores.json"


def normalize(s: str) -> str:
    """Collapse comma/space thousands separators so '75 000' == '75,000' == '75000'."""
    return re.sub(r"[,\s]", "", s.lower())


async def run_case(client, base_url, headers, case, top_k, doc_id_by_filename):
    r = await client.post(f"{base_url}/query", headers=headers,
                           json={"question": case["question"], "top_k": top_k}, timeout=60.0)
    r.raise_for_status()
    data = r.json()
    answer = data["answer"]
    sources = data.get("sources", [])
    debug = data.get("debug") or {}  # debug is explicitly null (not omitted) on early refusal

    top_chunks = debug.get("top_chunks", [])
    best_score = max((c.get("score", 0) for c in top_chunks), default=debug.get("best_rerank_score", 0))

    refused = answer.strip().lower().startswith(("i could not find", "i couldn't find", "no relevant information"))
    answered = not refused

    result = {
        "id": case["id"],
        "type": case["type"],
        "expect_answer": case["expect_answer"],
        "answered": answered,
        "abstention_correct": answered == case["expect_answer"],
        "best_score": best_score,
        "answer": answer[:300],
    }

    if case["expect_answer"]:
        expected_fname = case.get("expected_doc_filename")
        doc_ids_returned = [s["document"] for s in sources]  # already ranked by relevance_score desc
        result["n_sources"] = len(doc_ids_returned)

        if expected_fname:
            expected_doc_id = doc_id_by_filename.get(expected_fname)
            if expected_doc_id and expected_doc_id in doc_ids_returned:
                rank = doc_ids_returned.index(expected_doc_id) + 1  # 1-indexed
                result["recall_hit"] = True
                result["rank"] = rank
                result["reciprocal_rank"] = 1.0 / rank
            else:
                result["recall_hit"] = False
                result["rank"] = None
                result["reciprocal_rank"] = 0.0

        expected_substr = case.get("expected_substring")
        if expected_substr and answered:
            result["substring_correct"] = normalize(expected_substr) in normalize(answer)
        elif expected_substr:
            result["substring_correct"] = False
        else:
            result["substring_correct"] = None  # not checkable (e.g. case_summary)

    return result


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default=os.getenv("EVAL_API_URL", "http://localhost:8000"))
    ap.add_argument("--api-key", default=os.getenv("API_KEY", ""))
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    async with httpx.AsyncClient() as client:
        docs_resp = await client.get(f"{args.api_url}/documents", headers=headers, timeout=30.0)
        doc_id_by_filename = {d["filename"]: d["doc_id"] for d in docs_resp.json()["documents"]}

    results = []
    async with httpx.AsyncClient() as client:
        for i, case in enumerate(dataset):
            try:
                res = await run_case(client, args.api_url, headers, case, args.top_k, doc_id_by_filename)
            except Exception as e:
                res = {"id": case["id"], "type": case["type"], "expect_answer": case["expect_answer"],
                       "error": str(e)[:200], "abstention_correct": False}
            results.append(res)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(dataset)} cases done")

    # ── metrics ──
    n = len(results)
    abstention_correct = sum(1 for r in results if r.get("abstention_correct"))
    should_answer = [r for r in results if r["expect_answer"]]
    should_refuse = [r for r in results if not r["expect_answer"]]

    answered_when_should = sum(1 for r in should_answer if r.get("answered"))
    refused_when_should = sum(1 for r in should_refuse if not r.get("answered", True))

    checkable = [r for r in should_answer if r.get("substring_correct") is not None]
    correct_answers = sum(1 for r in checkable if r["substring_correct"])

    recall_cases = [r for r in should_answer if "recall_hit" in r]
    recall_hits = sum(1 for r in recall_cases if r["recall_hit"])
    mrr = sum(r["reciprocal_rank"] for r in recall_cases) / len(recall_cases) if recall_cases else 0

    print(f"\n{'='*60}")
    print(f"GOLDEN DATASET EVAL — {n} cases")
    print(f"{'='*60}")
    print(f"Recall@{args.top_k}: {recall_hits}/{len(recall_cases)} ({100*recall_hits/len(recall_cases):.1f}%)")
    print(f"MRR: {mrr:.3f}")
    print(f"Overall abstention accuracy: {abstention_correct}/{n} ({100*abstention_correct/n:.1f}%)")
    print(f"  - Answered when should ({len(should_answer)} cases):  {answered_when_should}/{len(should_answer)} ({100*answered_when_should/len(should_answer):.1f}%)")
    print(f"  - Refused when should ({len(should_refuse)} cases):   {refused_when_should}/{len(should_refuse)} ({100*refused_when_should/len(should_refuse):.1f}%)")
    print(f"Answer correctness (substring match, {len(checkable)} checkable cases): {correct_answers}/{len(checkable)} ({100*correct_answers/len(checkable):.1f}%)")

    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\n{len(errors)} cases errored:")
        for e in errors[:10]:
            print(f"  {e['id']}: {e['error']}")

    failures = [r for r in results if not r.get("abstention_correct")]
    if failures:
        print(f"\n{len(failures)} abstention failures:")
        for f in failures[:15]:
            print(f"  [{f['type']}] {f['id']}: expected_answer={f['expect_answer']} answered={f.get('answered')} score={f.get('best_score')}")

    wrong_answers = [r for r in checkable if not r["substring_correct"]]
    if wrong_answers:
        print(f"\n{len(wrong_answers)} wrong/missing-fact answers:")
        for w in wrong_answers[:15]:
            print(f"  {w['id']}: {w['answer'][:120]}")

    SCORES_OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved per-case scores to {SCORES_OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
