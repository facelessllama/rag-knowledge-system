#!/usr/bin/env python3
"""Joins the blind manual verdicts (blind_adjudication_verdicts.json —
filled in by reading blind_adjudication_titlefreescoped.json WITHOUT
looking at the key) with blind_adjudication_titlefreescoped_KEY.json to
reveal which model wrote each answer, then compares the manual-
adjudicated paired result against the automated judge's original
verdicts (from generator_ab_postfix_v3_titlefreescoped_results.json).

correct/partially_correct count as "correct" for the headline per-model
rate (partially_correct is reported separately too); wrong/refusal/
unsupported count as not-correct.

Usage:
    python eval/mixed_corpus/join_blind_adjudication.py \\
        --output eval/mixed_corpus/blind_adjudication_report.json
"""
import argparse
import json
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", default=str(CORPUS_DIR / "blind_adjudication_verdicts.json"))
    ap.add_argument("--key", default=str(CORPUS_DIR / "blind_adjudication_titlefreescoped_KEY.json"))
    ap.add_argument("--results", default=str(CORPUS_DIR / "generator_ab_postfix_v3_titlefreescoped_results.json"))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    verdicts = json.loads(Path(args.verdicts).read_text(encoding="utf-8"))
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    auto_results = {(r["id"], r["model_label"]): r for r in json.loads(Path(args.results).read_text(encoding="utf-8"))["results"]}

    per_model_rows = {}
    disagreements_with_judge = []
    for cid, letter_verdicts in verdicts.items():
        mapping = key.get(cid)
        if mapping is None:
            continue
        for letter, verdict in letter_verdicts.items():
            model = mapping.get(letter)
            if model is None:
                continue
            manual_correct = verdict in ("correct", "partially_correct")
            auto = auto_results.get((cid, model))
            auto_correct = auto["correct"] if auto else None
            per_model_rows.setdefault(model, []).append({
                "id": cid, "manual_verdict": verdict, "manual_correct": manual_correct,
                "auto_correct": auto_correct,
            })
            if auto_correct is not None and manual_correct != auto_correct:
                disagreements_with_judge.append({
                    "id": cid, "model": model, "manual_verdict": verdict,
                    "manual_correct": manual_correct, "auto_judge_correct": auto_correct,
                })

    summary = {}
    for model, rows in per_model_rows.items():
        n = len(rows)
        n_correct_strict = sum(1 for r in rows if r["manual_verdict"] == "correct")
        n_correct_lenient = sum(1 for r in rows if r["manual_correct"])
        n_partial = sum(1 for r in rows if r["manual_verdict"] == "partially_correct")
        n_wrong = sum(1 for r in rows if r["manual_verdict"] == "wrong")
        n_refusal = sum(1 for r in rows if r["manual_verdict"] == "refusal")
        n_auto_correct = sum(1 for r in rows if r["auto_correct"])
        summary[model] = {
            "n": n,
            "manual_correct_strict_pct": round(100 * n_correct_strict / n, 1),
            "manual_correct_or_partial_pct": round(100 * n_correct_lenient / n, 1),
            "n_correct_strict": n_correct_strict, "n_partially_correct": n_partial,
            "n_wrong": n_wrong, "n_refusal": n_refusal,
            "auto_judge_correct_pct": round(100 * n_auto_correct / n, 1),
        }

    output = {
        "summary": summary,
        "n_disagreements_with_automated_judge": len(disagreements_with_judge),
        "disagreements_with_judge": disagreements_with_judge,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'Model':<20} {'n':>4} {'manual strict':>14} {'manual+partial':>15} {'auto judge':>11}")
    for model, s in summary.items():
        print(f"{model:<20} {s['n']:>4} {s['manual_correct_strict_pct']:>13.1f}% "
              f"{s['manual_correct_or_partial_pct']:>14.1f}% {s['auto_judge_correct_pct']:>10.1f}%")
    print(f"\n{len(disagreements_with_judge)} disagreements between manual adjudication and the automated judge "
          f"(out of {sum(len(r) for r in per_model_rows.values())} answer judgments)")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
