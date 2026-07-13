#!/usr/bin/env python3
"""
Repeat-consistency test — the same question asked N times, to answer "does
this answer correctly once and then start hallucinating on repeats?" (see
eval/README.md, "Scale test: EU legislation corpus"). Complements
run_eval.py, which only measures single-shot accuracy per question.

Picks a representative sample from a golden dataset (spanning all
answerable types + a couple of adversarial refuse-cases), repeats each
--n-repeats times against a live /query, and for each repeat records not
just correct/wrong/refused (reusing run_eval.py's own run_case() grading,
so results are directly comparable) but *which document was actually
retrieved* — this separates retrieval flakiness (a different top document
across repeats) from generation flakiness (same retrieved context,
different answer), the same diagnostic split established for the A.B.A.
493 / Localism Act investigations on the original corpus (see
eval/README.md).

Usage:
    python eval/consistency_test.py --dataset eval/eu_golden_dataset.json \
        --api-url http://localhost:8001 --n-repeats 20 --n-questions 15 \
        --out eval/consistency_results_at_1000.json
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from dotenv import load_dotenv

load_dotenv()

from run_eval import normalize  # noqa: E402 — sibling module, see eval/build_heldout_dataset.py for the same pattern
from rag.generator import is_refusal

EVAL_DIR = Path(__file__).resolve().parent


def pick_sample(dataset: list[dict], n_questions: int) -> list[dict]:
    """One question per answerable type (round-robin) plus a couple of
    adversarial refuse-cases, up to n_questions total."""
    answerable = [c for c in dataset if c["expect_answer"]]
    adversarial = [c for c in dataset if not c["expect_answer"]]

    by_type: dict[str, list[dict]] = {}
    for c in answerable:
        by_type.setdefault(c["type"], []).append(c)

    n_adversarial = min(3, len(adversarial), max(1, n_questions // 5))
    sample = list(adversarial[:n_adversarial])

    types = list(by_type)
    i = 0
    while len(sample) < n_questions and any(by_type.values()):
        t = types[i % len(types)]
        if by_type[t]:
            sample.append(by_type[t].pop(0))
        i += 1
        if i > n_questions * 10:  # safety valve if types run dry
            break
    return sample[:n_questions]


async def grade_one(client, api_url, headers, case, top_k):
    """One /query call, graded the same way run_eval.py's run_case() does
    (answered/refused + substring correctness), plus which document was
    actually retrieved — a single round trip, not two."""
    r = await client.post(f"{api_url}/query", headers=headers,
                           json={"question": case["question"], "top_k": top_k}, timeout=60.0)
    r.raise_for_status()
    data = r.json()
    answer = data["answer"]
    sources = data.get("sources", [])
    answered = not is_refusal(answer)

    result = {
        "answered": answered,
        "abstention_correct": answered == case["expect_answer"],
        "retrieved_doc": sources[0]["document"] if sources else None,
    }
    expected_substr = case.get("expected_substring")
    if case["expect_answer"] and expected_substr:
        result["substring_correct"] = answered and normalize(expected_substr) in normalize(answer)
        result["abstention_correct"] = result["abstention_correct"] and result["substring_correct"]
    return result


async def repeat_case(client, api_url, headers, case, top_k, n_repeats):
    runs = []
    for _ in range(n_repeats):
        try:
            res = await grade_one(client, api_url, headers, case, top_k)
        except Exception as e:
            res = {"abstention_correct": False, "retrieved_doc": None, "error": str(e)[:200]}
        runs.append(res)
    return runs


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-url", default=os.getenv("EVAL_API_URL", "http://localhost:8001"))
    ap.add_argument("--api-key", default=os.getenv("API_KEY", ""))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--n-questions", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    sample = pick_sample(dataset, args.n_questions)
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    async with httpx.AsyncClient() as client:
        docs_resp = await client.get(f"{args.api_url}/documents", headers=headers, timeout=30.0)
        doc_id_by_filename = {d["filename"]: d["doc_id"] for d in docs_resp.json()["documents"]}

    print(f"Testing {len(sample)} questions x {args.n_repeats} repeats each "
          f"({len(sample) * args.n_repeats} total /query calls) ...")

    all_results = {}
    async with httpx.AsyncClient() as client:
        for i, case in enumerate(sample):
            runs = await repeat_case(client, args.api_url, headers, case, args.top_k, args.n_repeats)
            expected_doc_id = doc_id_by_filename.get(case.get("expected_doc_filename")) if case["expect_answer"] else None
            n_correct = sum(1 for r in runs if r.get("abstention_correct"))
            retrieved_docs = {r["retrieved_doc"] for r in runs}
            retrieval_stable = len(retrieved_docs) <= 1
            all_results[case["id"]] = {
                "question": case["question"],
                "type": case["type"],
                "expect_answer": case["expect_answer"],
                "correct_rate": f"{n_correct}/{args.n_repeats}",
                "retrieval_stable": retrieval_stable,
                "retrieved_docs_seen": list(retrieved_docs),
                "expected_doc_id": expected_doc_id,
                "runs": runs,
            }
            print(f"  [{i+1}/{len(sample)}] {case['type']:22s} {n_correct}/{args.n_repeats} correct | "
                  f"retrieval_stable={retrieval_stable} | {case['question'][:60]}")

    Path(args.out).write_text(json.dumps(all_results, indent=2, ensure_ascii=False))

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    perfect = sum(1 for r in all_results.values() if r["correct_rate"].split("/")[0] == r["correct_rate"].split("/")[1])
    unstable_retrieval = sum(1 for r in all_results.values() if not r["retrieval_stable"])
    print(f"Questions with 100% consistent (correct every repeat): {perfect}/{len(all_results)}")
    print(f"Questions with unstable retrieval (different top doc across repeats): {unstable_retrieval}/{len(all_results)}")
    for qid, r in all_results.items():
        flag = "" if r["correct_rate"].split("/")[0] == r["correct_rate"].split("/")[1] else "  <-- inconsistent"
        print(f"  {r['correct_rate']:>7} | retrieval_stable={str(r['retrieval_stable']):5} | {r['type']:22s} | {qid}{flag}")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
