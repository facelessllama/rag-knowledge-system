#!/usr/bin/env python3
"""Integration smoke test for the product's cloud-mode integration
(rag/generator.py::GeneratorRouter, wired into api/main.py) — NOT a
retrieval-quality eval. The standalone DeepSeek A/B (eval/compare_
generators.py) proves the MODEL is good; this proves the PRODUCT
actually calls it correctly end-to-end: real HTTP request -> real
retrieval -> real prompt assembly -> real provider dispatch -> a
sensible `provider`/`model` field back.

Picks a small sample of already-known confirmed-evidence-hit
fact_figure_caption cases, calls the LIVE /query endpoint twice per case
(provider="local" and provider="deepseek", unscoped, title-bearing —
same question wording as the standalone A/B for a fair comparison), and
reports:
- response['provider'] actually matches what was requested (integration
  sanity, not accuracy)
- per-case correctness, compared side-by-side against the standalone
  A/B's saved qwen2.5:7b/deepseek-v4-flash results for the SAME case IDs
  — this run does NOT need to match exactly (retrieval nondeterminism,
  different day/GPU state are expected — see project memory), only be in
  the same ballpark. A wildly different rate (e.g. deepseek near 0% here
  when the standalone A/B saw ~70-90%) would flag a real integration bug
  (wrong prompt shape, wrong model_id, provider field not actually
  switching anything), not a retrieval/generation quality question.

Usage:
    python eval/mixed_corpus/cloud_mode_integration_smoke.py \\
        --api-url http://localhost:8003 --n 15
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


async def call_query(client, api_url, headers, question, provider):
    r = await client.post(
        f"{api_url}/query", headers=headers,
        json={"question": question, "top_k": 5, "provider": provider},
        timeout=60.0,
    )
    r.raise_for_status()
    return r.json()


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden-v3", default=str(CORPUS_DIR / "golden_dataset_v3.json"))
    ap.add_argument("--postfix-results", default=str(CORPUS_DIR / "generator_ab_postfix_v3_results.json"),
                     help="Standalone A/B results to compare against, for the same case IDs")
    ap.add_argument("--postfix-contexts", default=str(CORPUS_DIR / "generator_ab_postfix_v3_contexts.json"),
                     help="Used only to pick case IDs already confirmed evidence-hit")
    ap.add_argument("--api-url", default=os.getenv("EVAL_API_URL", "http://localhost:8003"))
    ap.add_argument("--api-key", default=os.getenv("API_KEY", ""))
    ap.add_argument("--judge-url", default=os.getenv("OLLAMA_URL", "http://localhost:11435"))
    ap.add_argument("--judge-model", default=os.getenv("LLM_MODEL", "qwen2.5:7b"))
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20260722)
    args = ap.parse_args()

    golden_v3 = {c["id"]: c for c in json.loads(Path(args.golden_v3).read_text(encoding="utf-8"))}
    confirmed_ids = [c["id"] for c in json.loads(Path(args.postfix_contexts).read_text(encoding="utf-8"))["contexts"]]

    import random
    rng = random.Random(args.seed)
    sample_ids = rng.sample(confirmed_ids, min(args.n, len(confirmed_ids)))

    postfix_by_id_model = {
        (r["id"], r["model_label"]): r
        for r in json.loads(Path(args.postfix_results).read_text(encoding="utf-8"))["results"]
    }

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    rows = []
    async with httpx.AsyncClient() as client:
        judge_client = client
        for i, cid in enumerate(sample_ids):
            case = golden_v3[cid]
            question = case["question"]
            expected = case["expected_substring"]
            row = {"id": cid}
            for provider in ("local", "deepseek"):
                data = await call_query(client, args.api_url, headers, question, provider)
                answer = data["answer"]
                actual_provider = data.get("provider")
                sub_ok = substring_ok(expected, normalize(answer)) if expected else None
                sem_ok = None
                if expected and not sub_ok:
                    sem_ok = await judge_semantic_match(
                        judge_client, args.judge_url, args.judge_model, question, expected, answer
                    )
                row[provider] = {
                    "requested_provider": provider,
                    "actual_provider": actual_provider,
                    "provider_matches": actual_provider == provider,
                    "model": data.get("model"),
                    "correct": bool(sub_ok or sem_ok),
                    "answer": answer[:200],
                }
            rows.append(row)
            print(f"[{i+1}/{len(sample_ids)}] {cid}: "
                  f"local={'OK' if row['local']['correct'] else 'wrong'} "
                  f"(provider={row['local']['actual_provider']}) | "
                  f"deepseek={'OK' if row['deepseek']['correct'] else 'wrong'} "
                  f"(provider={row['deepseek']['actual_provider']})", flush=True)

    n = len(rows)
    provider_field_correct = sum(1 for r in rows for p in ("local", "deepseek") if r[p]["provider_matches"])
    local_correct = sum(1 for r in rows if r["local"]["correct"])
    deepseek_correct = sum(1 for r in rows if r["deepseek"]["correct"])

    # side-by-side against the standalone A/B, same case IDs
    standalone_local = [postfix_by_id_model.get((r["id"], "qwen2.5:7b")) for r in rows]
    standalone_cloud = [postfix_by_id_model.get((r["id"], "deepseek-v4-flash")) for r in rows]
    standalone_local_correct = sum(1 for r in standalone_local if r and r["correct"])
    standalone_cloud_correct = sum(1 for r in standalone_cloud if r and r["correct"])

    print(f"\n{'='*60}")
    print(f"n={n} sample cases")
    print(f"provider field correctness: {provider_field_correct}/{n*2} "
          f"(response['provider'] matched what was requested)")
    print(f"\n{'Path':<30} {'local/qwen':>12} {'deepseek/flash':>16}")
    print(f"{'Live /query endpoint':<30} {local_correct:>9}/{n}  {deepseek_correct:>13}/{n}")
    print(f"{'Standalone A/B (same IDs)':<30} {standalone_local_correct:>9}/{n}  {standalone_cloud_correct:>13}/{n}")
    print(f"{'='*60}")
    print(
        "\nThis is an integration sanity check, not a fresh accuracy measurement — "
        "expect noise, not an exact match, between the two rows above. A live-endpoint "
        "rate wildly below the standalone row (e.g. 0-1/N when standalone is 10+/N) "
        "would flag a real integration bug, not just retrieval/generation variance."
    )

    output = {
        "n": n,
        "provider_field_correct": provider_field_correct,
        "live_local_correct": local_correct,
        "live_deepseek_correct": deepseek_correct,
        "standalone_local_correct": standalone_local_correct,
        "standalone_cloud_correct": standalone_cloud_correct,
        "rows": rows,
    }
    out_path = CORPUS_DIR / "cloud_mode_integration_smoke_report.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
