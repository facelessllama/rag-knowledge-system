#!/usr/bin/env python3
"""Re-scores ALREADY-SAVED generator answers (eval/mixed_corpus/
generator_ab_results.json — qwen2.5:7b/deepseek-v4-flash/deepseek-v4-pro,
captured against golden-dataset v2's expected_substring) against golden-
dataset v3's caption_text instead — WITHOUT calling any generator model
again. The only model call this script makes is the same local semantic-
judge re-verification run_eval.py/compare_generators.py already use on a
substring miss, never a new generation.

Produces the paired report the v3 plan calls for:
- how many answers flip wrong->correct or correct->wrong under v3
- how many remain wrong under both (still all-wrong)
- which flips are attributable to a genuine label fix (the case's
  label_status is "extracted"/"ambiguous" — its expected_substring
  actually changed) vs. pure scorer noise (label_status "verified" — v3's
  expected_substring is identical to v2's, so a flip there can only be
  semantic-judge nondeterminism, never a real fix)
- per-model conditional correctness under v2 vs v3, same n throughout
  (all rows are on the pre-selected confirmed-evidence-chunk sample
  already, so this already IS "same evidence-hit set" scoring)

Usage:
    python eval/mixed_corpus/rescore_against_v3.py \\
        --results eval/mixed_corpus/generator_ab_results.json \\
        --golden-v3 eval/mixed_corpus/golden_dataset_v3.json \\
        --output eval/mixed_corpus/rescore_v3_report.json
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

from run_eval import normalize, substring_ok, judge_semantic_match

CORPUS_DIR = Path(__file__).resolve().parent


async def rescore_row(client, row, v3_case, judge_url, judge_model):
    """Re-derives correctness for one saved answer against v3's
    expected_substring — same logic as run_eval.py's run_case(), applied
    to an already-generated answer instead of a fresh one."""
    answer = row["answer"]
    expected_v3 = v3_case["expected_substring"]
    if row.get("refused"):
        return {"substring_correct_v3": False, "semantic_correct_v3": False, "correct_v3": False}
    substring_correct_v3 = substring_ok(expected_v3, normalize(answer))
    semantic_correct_v3 = False
    if not substring_correct_v3:
        semantic_correct_v3 = await judge_semantic_match(
            client, judge_url, judge_model, v3_case["question"], expected_v3, answer
        )
    return {
        "substring_correct_v3": substring_correct_v3,
        "semantic_correct_v3": semantic_correct_v3,
        "correct_v3": substring_correct_v3 or semantic_correct_v3,
    }


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(CORPUS_DIR / "generator_ab_results.json"))
    ap.add_argument("--golden-v3", default=str(CORPUS_DIR / "golden_dataset_v3.json"))
    ap.add_argument("--judge-url", default=os.getenv("OLLAMA_URL", "http://localhost:11435"))
    ap.add_argument("--judge-model", default=os.getenv("LLM_MODEL", "qwen2.5:7b"))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    saved = json.loads(Path(args.results).read_text(encoding="utf-8"))
    results = saved["results"]
    golden_v3 = {c["id"]: c for c in json.loads(Path(args.golden_v3).read_text(encoding="utf-8"))}

    rows = []
    async with httpx.AsyncClient() as client:
        for i, row in enumerate(results):
            v3_case = golden_v3.get(row["id"])
            if v3_case is None:
                continue  # shouldn't happen — same seed, same case IDs
            rescored = await rescore_row(client, row, v3_case, args.judge_url, args.judge_model)
            merged = {**row, **rescored, "label_status": v3_case.get("label_status")}
            rows.append(merged)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(results)} done", flush=True)

    # ── per-model, per-transition breakdown ──
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model_label"], []).append(r)

    summary = {}
    for model, model_rows in by_model.items():
        n = len(model_rows)
        v2_correct = sum(1 for r in model_rows if r["correct"])
        v3_correct = sum(1 for r in model_rows if r["correct_v3"])
        wrong_to_correct = [r for r in model_rows if not r["correct"] and r["correct_v3"]]
        correct_to_wrong = [r for r in model_rows if r["correct"] and not r["correct_v3"]]
        still_wrong = [r for r in model_rows if not r["correct"] and not r["correct_v3"]]
        still_correct = [r for r in model_rows if r["correct"] and r["correct_v3"]]

        def _attributable(subset):
            return sum(1 for r in subset if r["label_status"] in ("extracted", "ambiguous"))

        summary[model] = {
            "n": n,
            "v2_correct": v2_correct, "v2_correct_pct": round(100 * v2_correct / n, 1),
            "v3_correct": v3_correct, "v3_correct_pct": round(100 * v3_correct / n, 1),
            "wrong_to_correct": len(wrong_to_correct),
            "wrong_to_correct_attributable_to_label_fix": _attributable(wrong_to_correct),
            "correct_to_wrong": len(correct_to_wrong),
            "correct_to_wrong_attributable_to_label_fix": _attributable(correct_to_wrong),
            "still_all_wrong": len(still_wrong),
            "still_all_wrong_on_verified_label": sum(1 for r in still_wrong if r["label_status"] == "verified"),
            "unchanged_correct": len(still_correct),
        }

    output = {
        "run_metadata": {
            "results_source": args.results,
            "golden_v3_source": args.golden_v3,
            "n_rows": len(rows),
        },
        "summary": summary,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'Model':<20} {'n':>4} {'v2 correct':>12} {'v3 correct':>12} {'wrong->correct':>15} {'correct->wrong':>15} {'still all-wrong':>16}")
    for model, s in summary.items():
        print(f"{model:<20} {s['n']:>4} {s['v2_correct_pct']:>10.1f}% {s['v3_correct_pct']:>10.1f}% "
              f"{s['wrong_to_correct']:>15} {s['correct_to_wrong']:>15} {s['still_all_wrong']:>16}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
