#!/usr/bin/env python3
"""Prepares an ANONYMIZED file for blind manual adjudication of every
title-free-scoped disagreement case (where the automated judge's verdict
differs across qwen2.5:7b/deepseek-v4-flash/deepseek-v4-pro) — per the
user's instruction that the manual-adjudicated paired result, not the
automatic judge, should be the basis for any generator decision.

For each case: model labels are shuffled to A/B/C with a per-case random
seed (so "A" isn't always the same model across cases), and NEITHER the
model name NOR the automated judge's verdict is included in the output —
only question, expected fact, and the three answers. A separate,
NOT-anonymized key file maps case_id -> {A: model_label, ...} so verdicts
can be re-joined to real model identities afterward without the
adjudicator (a human, or here, a separate blind reading pass) ever
seeing that mapping while judging.

Usage:
    python eval/mixed_corpus/prepare_blind_adjudication.py \\
        --results eval/mixed_corpus/generator_ab_postfix_v3_titlefreescoped_results.json \\
        --golden-v3 eval/mixed_corpus/golden_dataset_v3.json \\
        --output eval/mixed_corpus/blind_adjudication_titlefreescoped.json \\
        --key eval/mixed_corpus/blind_adjudication_titlefreescoped_KEY.json
"""
import argparse
import json
import random
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--golden-v3", default=str(CORPUS_DIR / "golden_dataset_v3.json"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--seed", type=int, default=20260722)
    args = ap.parse_args()

    raw = json.loads(Path(args.results).read_text(encoding="utf-8"))
    results = raw["results"]
    golden_v3 = {c["id"]: c for c in json.loads(Path(args.golden_v3).read_text(encoding="utf-8"))}

    by_id_model = {}
    labels = []
    for r in results:
        by_id_model[(r["id"], r["model_label"])] = r
        if r["model_label"] not in labels:
            labels.append(r["model_label"])

    case_ids = sorted({r["id"] for r in results})
    disagreement_ids = []
    for cid in case_ids:
        entries = [by_id_model.get((cid, m)) for m in labels]
        if any(e is None for e in entries):
            continue
        corrects = {e["correct"] for e in entries}
        if len(corrects) > 1:
            disagreement_ids.append(cid)

    rng = random.Random(args.seed)
    blind_items = []
    key = {}
    for cid in disagreement_ids:
        case = golden_v3.get(cid, {})
        shuffled = labels[:]
        rng.shuffle(shuffled)
        letters = ["A", "B", "C", "D", "E"][: len(shuffled)]
        mapping = dict(zip(letters, shuffled))
        key[cid] = mapping
        blind_items.append({
            "id": cid,
            "question": case.get("question"),
            "expected_fact": case.get("expected_substring"),
            "label_status": case.get("label_status"),
            "answers": {letter: by_id_model[(cid, model)]["answer"] for letter, model in mapping.items()},
            "verdict": None,  # to be filled in: correct | partially_correct | wrong | refusal | unsupported, per letter
        })

    Path(args.output).write_text(json.dumps({
        "instructions": (
            "For each case, read the question and expected_fact, then judge EACH "
            "answer (A/B/C) independently as one of: correct, partially_correct, "
            "wrong, refusal, unsupported (answer makes a claim the evidence doesn't "
            "support). Do not compare answers against each other's wording style — "
            "judge each against expected_fact and plausibility given the question. "
            "Fill in 'verdict' as {'A': '...', 'B': '...', 'C': '...'}."
        ),
        "n_cases": len(blind_items),
        "items": blind_items,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.key).write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{len(disagreement_ids)} disagreement cases written to {args.output} (anonymized)")
    print(f"Key (NOT for use while adjudicating) written to {args.key}")


if __name__ == "__main__":
    main()
