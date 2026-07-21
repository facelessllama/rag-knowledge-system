#!/usr/bin/env python3
"""Analyzes eval/compare_generators.py's post-fix, golden-v3-scored A/B
(eval/mixed_corpus/generator_ab_postfix_v3_results.json) — joins raw
per-(case, model) results with golden_dataset_v3.json's label_status and
computes the three metrics the post-fix A/B plan calls for, plus a
model-agreement transition breakdown and manual-review candidate lists.

Metrics (per model, per label-status slice — "verified" only is the
headline, "verified"+"extracted" a second cut, "ambiguous" reported
separately, never folded into either headline):
- Conditional correctness: correct / n, over cases with a confirmed
  evidence_chunk_hit only (the population compare_generators.py actually
  ran generation on).
- End-to-end supported correctness: (evidence_hit AND correct) / ALL
  eligible fact_figure_caption cases in that label-status slice
  (including the ones excluded for no evidence_chunk_hit at all) —
  computed directly per-case, never by multiplying rounded percentages.
- Refused despite evidence: refused / n, over the same evidence-hit
  population.

Usage:
    python eval/mixed_corpus/analyze_postfix_ab.py \\
        --results eval/mixed_corpus/generator_ab_postfix_v3_results.json \\
        --golden-v3 eval/mixed_corpus/golden_dataset_v3.json \\
        --output eval/mixed_corpus/analyze_postfix_ab_report.json
"""
import argparse
import json
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent
# "verified_reviewed" = a plain "verified" case that was additionally
# hand-checked against the real source PDF (see apply_manual_label_
# review.py) and confirmed genuinely clean — counts as verified, not a
# separate tier; it exists so the audit trail records WHICH verified
# cases were actually read by a human, not just auto-classified.
_VERIFIED_STATUSES = ("verified", "verified_reviewed")
LABEL_TIERS = {
    "verified_only": lambda ls: ls in _VERIFIED_STATUSES,
    "verified_plus_extracted": lambda ls: ls in _VERIFIED_STATUSES + ("extracted",),
}


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(CORPUS_DIR / "generator_ab_postfix_v3_results.json"))
    ap.add_argument("--golden-v3", default=str(CORPUS_DIR / "golden_dataset_v3.json"))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    raw = json.loads(Path(args.results).read_text(encoding="utf-8"))
    golden_v3 = {c["id"]: c for c in json.loads(Path(args.golden_v3).read_text(encoding="utf-8"))}
    all_caption_ids = [cid for cid, c in golden_v3.items() if c["type"] == "fact_figure_caption"]

    results = raw["results"]
    by_id_model: dict[tuple[str, str], dict] = {}
    labels = []
    for r in results:
        by_id_model[(r["id"], r["model_label"])] = r
        if r["model_label"] not in labels:
            labels.append(r["model_label"])

    captured_ids = sorted({r["id"] for r in results})  # cases compare_generators.py actually ran (evidence-hit)

    def label_status_of(cid):
        return golden_v3.get(cid, {}).get("label_status")

    # ── metrics per model, per tier (verified_only / verified_plus_extracted), plus ambiguous reported separately ──
    metrics = {}
    for model in labels:
        model_metrics = {}
        for tier_name, tier_pred in LABEL_TIERS.items():
            tier_all_ids = [cid for cid in all_caption_ids if tier_pred(label_status_of(cid))]
            tier_captured_ids = [cid for cid in captured_ids if tier_pred(label_status_of(cid))]
            entries = [by_id_model[(cid, model)] for cid in tier_captured_ids if (cid, model) in by_id_model]
            n_conditional = len(entries)
            n_correct = sum(1 for e in entries if e["correct"])
            n_refused = sum(1 for e in entries if e["refused"])
            n_end_to_end_supported = sum(
                1 for cid in tier_all_ids
                if (cid, model) in by_id_model and by_id_model[(cid, model)]["correct"]
            )
            model_metrics[tier_name] = {
                "n_all_eligible": len(tier_all_ids),
                "n_evidence_hit": n_conditional,
                "conditional_correctness": {"n": n_correct, "of": n_conditional, "pct": pct(n_correct, n_conditional)},
                "end_to_end_supported_correctness": {
                    "n": n_end_to_end_supported, "of": len(tier_all_ids),
                    "pct": pct(n_end_to_end_supported, len(tier_all_ids)),
                },
                "refused_despite_evidence": {"n": n_refused, "of": n_conditional, "pct": pct(n_refused, n_conditional)},
            }
        # ambiguous reported separately, never in a headline tier
        ambiguous_ids = [cid for cid in all_caption_ids if label_status_of(cid) == "ambiguous"]
        ambiguous_captured = [cid for cid in captured_ids if label_status_of(cid) == "ambiguous"]
        amb_entries = [by_id_model[(cid, model)] for cid in ambiguous_captured if (cid, model) in by_id_model]
        model_metrics["ambiguous_(not_headline)"] = {
            "n_all_eligible": len(ambiguous_ids),
            "n_evidence_hit": len(amb_entries),
            "conditional_correctness": {
                "n": sum(1 for e in amb_entries if e["correct"]), "of": len(amb_entries),
                "pct": pct(sum(1 for e in amb_entries if e["correct"]), len(amb_entries)),
            },
            "refused_despite_evidence": {
                "n": sum(1 for e in amb_entries if e["refused"]), "of": len(amb_entries),
                "pct": pct(sum(1 for e in amb_entries if e["refused"]), len(amb_entries)),
            },
        }
        metrics[model] = model_metrics

    # ── model-agreement transitions (over ALL captured/evidence-hit cases, all label statuses, headline filtering is metrics' job) ──
    qwen_label = next((l for l in labels if "qwen" in l.lower()), labels[0])
    other_labels = [l for l in labels if l != qwen_label]

    pairwise = {}
    for other in other_labels:
        qwen_wrong_other_correct = []
        qwen_correct_other_wrong = []
        both_correct = []
        both_wrong = []
        for cid in captured_ids:
            eq, eo = by_id_model.get((cid, qwen_label)), by_id_model.get((cid, other))
            if eq is None or eo is None:
                continue
            if not eq["correct"] and eo["correct"]:
                qwen_wrong_other_correct.append(cid)
            elif eq["correct"] and not eo["correct"]:
                qwen_correct_other_wrong.append(cid)
            elif eq["correct"] and eo["correct"]:
                both_correct.append(cid)
            else:
                both_wrong.append(cid)
        pairwise[f"{qwen_label}_vs_{other}"] = {
            "qwen_wrong_other_correct": len(qwen_wrong_other_correct),
            "qwen_correct_other_wrong": len(qwen_correct_other_wrong),
            "both_correct": len(both_correct),
            "both_wrong": len(both_wrong),
            "qwen_correct_other_wrong_ids": qwen_correct_other_wrong,
        }

    all_three_correct = []
    all_three_wrong = []
    disagreement_ids = []
    for cid in captured_ids:
        entries = [by_id_model.get((cid, m)) for m in labels]
        if any(e is None for e in entries):
            continue
        corrects = [e["correct"] for e in entries]
        if all(corrects):
            all_three_correct.append(cid)
        elif not any(corrects):
            all_three_wrong.append(cid)
        else:
            disagreement_ids.append(cid)

    # ── manual-review candidate lists (not auto-judged further — read by hand) ──
    def case_snippet(cid):
        out = {"id": cid, "label_status": label_status_of(cid),
               "expected_substring": golden_v3.get(cid, {}).get("expected_substring")}
        for m in labels:
            e = by_id_model.get((cid, m))
            if e:
                out[m] = {"correct": e["correct"], "refused": e["refused"], "answer": e["answer"][:300]}
        return out

    manual_review = {
        "all_disagreements": [case_snippet(cid) for cid in disagreement_ids],
        "qwen_correct_deepseek_wrong": {
            other: [case_snippet(cid) for cid in pairwise[f"{qwen_label}_vs_{other}"]["qwen_correct_other_wrong_ids"]]
            for other in other_labels
        },
        "all_three_wrong_sample": [case_snippet(cid) for cid in all_three_wrong[:10]],
    }

    output = {
        "run_metadata": {
            "results_source": args.results,
            "golden_v3_source": args.golden_v3,
            "n_captured_cases": len(captured_ids),
            "n_all_eligible_cases": len(all_caption_ids),
            "models": labels,
        },
        "metrics_by_model": metrics,
        "pairwise_transitions": pairwise,
        "three_way": {
            "all_correct": len(all_three_correct), "all_wrong": len(all_three_wrong),
            "disagreement": len(disagreement_ids),
        },
        "manual_review": manual_review,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(captured_ids)}/{len(all_caption_ids)} cases had a confirmed evidence_chunk_hit\n")
    for model in labels:
        print(f"=== {model} ===")
        for tier in ("verified_only", "verified_plus_extracted", "ambiguous_(not_headline)"):
            m = metrics[model][tier]
            cc = m["conditional_correctness"]
            print(f"  {tier:28s} conditional={cc['n']}/{cc['of']} ({cc['pct']}%)", end="")
            if "end_to_end_supported_correctness" in m:
                e = m["end_to_end_supported_correctness"]
                print(f"  end-to-end={e['n']}/{e['of']} ({e['pct']}%)", end="")
            rd = m["refused_despite_evidence"]
            print(f"  refused_despite_evidence={rd['n']}/{rd['of']} ({rd['pct']}%)")
    print(f"\nThree-way: all_correct={len(all_three_correct)} all_wrong={len(all_three_wrong)} disagreement={len(disagreement_ids)}")
    for k, v in pairwise.items():
        print(f"{k}: qwen_wrong->other_correct={v['qwen_wrong_other_correct']} "
              f"qwen_correct->other_wrong={v['qwen_correct_other_wrong']}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
