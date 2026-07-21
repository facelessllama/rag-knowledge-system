#!/usr/bin/env python3
"""Generic, corpus-agnostic multi-model generation comparison tool.

Takes a FROZEN context file (see eval/mixed_corpus/capture_generation_
contexts.py, or any producer following the same
{"contexts": [{"id", "messages", "expected_substring", ...}, ...]} shape)
and runs each requested model against the IDENTICAL `messages` for every
case. This tool never touches Qdrant, embeddings, or the reranker — it
exists specifically to isolate GENERATION quality from RETRIEVAL
quality, which is why capturing the context is a separate, one-time step
instead of something this script does itself.

Resumable: (case_id, model_label) pairs already recorded in
--progress-path are skipped on a re-run, so an interrupted comparison
(rate limit, network blip, a very long context set) can continue rather
than re-spending API-metered generation calls that already succeeded.

Grading uses a LOCAL judge model only (--judge-provider/--judge-model,
default ollama/qwen2.5:7b) — deliberately never one of the cloud models
under comparison, so no model ever grades its own or a competitor's
answer.

API keys are read ONLY from environment variables (DEEPSEEK_API_KEY,
...) — never accepted as a CLI flag, to keep them out of shell history
and process listings.

Usage:
    python eval/compare_generators.py \\
        --contexts /tmp/contexts.json \\
        --model ollama/qwen2.5:7b \\
        --model deepseek/deepseek-v4-flash \\
        --model deepseek/deepseek-v4-pro \\
        --progress-path eval/mixed_corpus/generator_ab_progress.json \\
        --output eval/mixed_corpus/generator_ab_results.json \\
        --report eval/mixed_corpus/generator_ab_report.md
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from dotenv import load_dotenv
load_dotenv()

from rag.generator import LLMGenerator, DeepSeekGenerator, is_refusal
# Reused rather than duplicated — same normalize()/substring_ok()/
# judge_semantic_match() eval/run_eval.py already has and tests indirectly;
# keeping one copy avoids the two drifting apart.
from run_eval import normalize, substring_ok, judge_semantic_match
import httpx


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


class ModelSpec:
    """provider/model_id[=label] -> a generator + a stable label used as
    the key in results/progress/report tables. Label defaults to
    provider/model_id when not given explicitly."""

    def __init__(self, spec: str, ollama_url: str, deepseek_api_key: str | None):
        explicit_label = None
        if "=" in spec:
            spec, explicit_label = spec.split("=", 1)
        provider, _, model_id = spec.partition("/")
        if not model_id:
            raise ValueError(f"--model must be provider/model_id (e.g. ollama/qwen2.5:7b), got: {spec!r}")
        self.label = explicit_label or model_id  # default label: just the model_id, not "provider/model_id" — reads cleaner in report tables
        self.provider = provider
        self.model_id = model_id
        if provider == "ollama":
            self.generator = LLMGenerator(ollama_url=ollama_url, model=model_id)
        elif provider == "deepseek":
            if not deepseek_api_key:
                raise SystemExit(
                    "DEEPSEEK_API_KEY is not set in the environment — required for any --model deepseek/... "
                    "(API keys are read from env only, never a CLI flag)."
                )
            self.generator = DeepSeekGenerator(api_key=deepseek_api_key, model=model_id)
        else:
            raise ValueError(f"Unknown provider {provider!r} (known: ollama, deepseek)")

    async def generate(self, messages: list[dict]) -> dict:
        return await self.generator.generate(messages)


def load_progress(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_progress(path: Path, progress: dict):
    path.write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contexts", required=True, help="Context file from capture_generation_contexts.py")
    ap.add_argument("--model", action="append", required=True, dest="models",
                     help="provider/model_id[=label], repeatable. providers: ollama, deepseek")
    ap.add_argument("--judge-provider-url", default=os.getenv("OLLAMA_URL", "http://localhost:11435"))
    ap.add_argument("--judge-model", default="qwen2.5:7b",
                     help="Local judge model — never one of the cloud models under comparison")
    ap.add_argument("--progress-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", default=None, help="Optional Markdown summary report path")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--breakdown-by", default=None,
                     help="Dotted meta field (e.g. meta.saved_by_change) — if truthy on a context, "
                          "an additional aggregate table is emitted for just that subset alongside "
                          "the full-set one.")
    args = ap.parse_args()

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11435")
    specs = [ModelSpec(m, ollama_url, deepseek_api_key) for m in args.models]
    labels = [s.label for s in specs]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"Duplicate --model labels: {labels}")

    ctx_data = json.loads(Path(args.contexts).read_text(encoding="utf-8"))
    contexts = ctx_data["contexts"]
    if args.limit:
        contexts = contexts[: args.limit]
    print(f"{len(contexts)} contexts x {len(specs)} models = {len(contexts) * len(specs)} (case, model) pairs", flush=True)

    progress_path = Path(args.progress_path)
    progress = load_progress(progress_path)  # "{case_id}::{label}" -> result dict

    async with httpx.AsyncClient() as judge_client:
        for i, ctx in enumerate(contexts):
            cid = ctx["id"]
            for spec in specs:
                key = f"{cid}::{spec.label}"
                if key in progress:
                    continue
                result = await spec.generate(ctx["messages"])
                answer = result["answer"]
                refused = is_refusal(answer)
                expected = ctx.get("expected_substring")
                grade = {"refused": refused, "correct": False, "substring_correct": None, "semantic_correct": None}
                if not refused and expected:
                    norm = normalize(answer)
                    sub_ok = substring_ok(expected, norm)
                    grade["substring_correct"] = sub_ok
                    sem_ok = None
                    if not sub_ok:
                        sem_ok = await judge_semantic_match(
                            judge_client, args.judge_provider_url, args.judge_model,
                            ctx["question"], expected, answer,
                        )
                        grade["semantic_correct"] = sem_ok
                    grade["correct"] = bool(sub_ok or sem_ok)
                progress[key] = {
                    "id": cid, "model_label": spec.label, "provider": spec.provider, "model_id": spec.model_id,
                    "answer": answer, "tokens": result.get("total_tokens", 0), **grade,
                }
                save_progress(progress_path, progress)
                print(f"[{i+1}/{len(contexts)}] {cid:45s} {spec.label:22s} "
                      f"{'REFUSED' if refused else ('OK' if grade['correct'] else 'wrong')}", flush=True)

    def pct(a, b):
        return f"{a}/{b} ({100*a/b:.1f}%)" if b else "n/a"

    by_id_model = {(e["id"], e["model_label"]): e for e in progress.values()}

    def build_report_for(ids: list[str], heading: str) -> list[str]:
        lines = [f"## {heading} (n={len(ids)})", "",
                 "| model | correct | refused_despite_context | avg tokens |", "|---|---|---|---|"]
        for label in labels:
            entries = [by_id_model[(cid, label)] for cid in ids if (cid, label) in by_id_model]
            n_correct = sum(1 for e in entries if e["correct"])
            n_refused = sum(1 for e in entries if e["refused"])
            avg_tokens = sum(e["tokens"] for e in entries) / len(entries) if entries else 0
            lines.append(f"| {label} | {pct(n_correct, len(entries))} | {pct(n_refused, len(entries))} | {avg_tokens:.0f} |")

        lines.append("")
        lines.append("### Pairwise (2x2 correctness contingency)")
        for a_idx in range(len(labels)):
            for b_idx in range(a_idx + 1, len(labels)):
                a_label, b_label = labels[a_idx], labels[b_idx]
                both_correct = a_only = b_only = both_wrong = 0
                for cid in ids:
                    ea, eb = by_id_model.get((cid, a_label)), by_id_model.get((cid, b_label))
                    if ea is None or eb is None:
                        continue
                    if ea["correct"] and eb["correct"]:
                        both_correct += 1
                    elif ea["correct"] and not eb["correct"]:
                        a_only += 1
                    elif eb["correct"] and not ea["correct"]:
                        b_only += 1
                    else:
                        both_wrong += 1
                lines.append(f"\n#### {a_label} vs {b_label}\n")
                lines.append(f"| | {b_label} correct | {b_label} wrong |")
                lines.append(f"|---|---|---|")
                lines.append(f"| **{a_label} correct** | {both_correct} | {a_only} |")
                lines.append(f"| **{a_label} wrong** | {b_only} | {both_wrong} |")
        return lines

    all_ids = [c["id"] for c in contexts]
    report_sections = build_report_for(all_ids, "All cases")

    if args.breakdown_by:
        _, _, field = args.breakdown_by.partition(".")
        field = field or args.breakdown_by
        subset_ids = [c["id"] for c in contexts if c.get("meta", {}).get(field)]
        if subset_ids:
            report_sections += [""] + build_report_for(subset_ids, f"Subset: {args.breakdown_by}")
        else:
            print(f"--breakdown-by {args.breakdown_by}: no contexts had this field truthy, skipping subset table")

    print("\n" + "\n".join(report_sections))

    Path(args.output).write_text(json.dumps({
        "run_metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "contexts_file": args.contexts,
            "contexts_run_metadata": ctx_data.get("run_metadata"),
            "models": [{"label": s.label, "provider": s.provider, "model_id": s.model_id} for s in specs],
            "judge_model": args.judge_model,
            "n_contexts": len(contexts),
        },
        "results": list(progress.values()),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.output}")

    if args.report:
        report = [
            f"# Generator comparison — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            "",
            f"Contexts: `{args.contexts}` ({len(contexts)} cases, git `{git_commit()[:8]}`)",
            f"Judge: `{args.judge_model}` (local, never one of the compared models)",
            "",
            *report_sections,
        ]
        Path(args.report).write_text("\n".join(report) + "\n", encoding="utf-8")
        print(f"Wrote {args.report}")


if __name__ == "__main__":
    asyncio.run(main())
