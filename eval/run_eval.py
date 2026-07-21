#!/usr/bin/env python3
"""
Runs an eval dataset (eval/golden_dataset.json by default, or any dataset
built by build_golden_dataset.py / build_heldout_dataset.py) against a live
instance and reports:

- Recall@K (was the expected document among the K sources returned?)
- MRR (mean reciprocal rank of the expected document among sources)
- abstention accuracy (did the system correctly answer vs. correctly refuse?)
- answer correctness (does the answer contain the expected fact substring?
  pass --semantic-judge to fall back to a local-LLM judge on substring
  misses only — see judge_semantic_match()'s docstring for why this exists)

Also dumps every case's retrieval score and rerank score to
eval/last_run_scores_<dataset name>.json — the golden_dataset.json run is
what RELEVANCE_THRESHOLD (see api/main.py) was calibrated from. Run against
heldout_dataset.json (documents + adversarial questions never touched
during that calibration) to check the threshold and fixes generalize
rather than just fitting the set they were tuned on.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --dataset eval/heldout_dataset.json
    python eval/run_eval.py --api-url http://localhost:8000 --top-k 5
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
import os

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.generator import is_refusal

EVAL_DIR = Path(__file__).resolve().parent


def normalize(s: str) -> str:
    """Collapse comma/space thousands separators so '75 000' == '75,000' == '75000'."""
    return re.sub(r"[,\s]", "", s.lower())


def substring_ok(expected, answer_norm: str) -> bool:
    """expected is either a single fact string or a list of equally-valid
    fact strings (e.g. cross_reference_fact/semantic_cross_reference,
    where a document can legitimately cite more than one earlier
    regulation — see eval/build_eu_golden_dataset.py::extract_facts).
    Matching any one of them counts as correct."""
    candidates = expected if isinstance(expected, list) else [expected]
    return any(normalize(c) in answer_norm for c in candidates)


def evidence_chunk_overlaps(chunk_start, chunk_end, cap_start, cap_end, min_coverage: float = 0.5) -> bool:
    """True if a chunk's char range gives meaningful coverage of a
    caption's char range — both are offsets into the SAME normalize_
    whitespace() coordinate space (see ingestion/chunker.py's char_start/
    char_end and eval/mixed_corpus/build_golden_dataset.py's evidence_
    char_start/char_end), so they can be intersected directly with no
    text comparison needed.

    Deliberately NOT "any intersection at all" — a chunk overlapping the
    caption by one character technically intersects but tells us nothing
    (chunk_size=512 means an adjacent chunk routinely brushes a caption's
    edge by a few chars of shared context). Counts as a hit if either (a)
    the chunk contains the caption's OPENING character — the strongest
    single signal, since the model at least saw where the caption starts —
    or (b) the overlap covers at least `min_coverage` of the caption's own
    span, for a caption long enough to legitimately span more than one
    chunk."""
    overlap = max(0, min(chunk_end, cap_end) - max(chunk_start, cap_start))
    if overlap == 0:
        return False
    contains_caption_start = chunk_start <= cap_start < chunk_end
    cap_len = cap_end - cap_start
    covers_enough = cap_len > 0 and overlap / cap_len >= min_coverage
    return contains_caption_start or covers_enough


_JUDGE_PROMPT = """You are grading whether a generated answer correctly conveys a specific expected fact. Paraphrasing, reordering, or different wording is fine — the core fact just has to be present and correct. Do not give credit for a vague, partial, hedged, or wrong answer.

Question: {question}
Expected fact: {expected}
Generated answer: {answer}

Does the generated answer correctly convey the expected fact? Reply with exactly one word: YES or NO."""


async def judge_semantic_match(client, ollama_url: str, model: str, question: str, expected: str, answer: str) -> bool:
    """Fallback for when substring_ok() says no — a strict verbatim check is
    the right default (see eval/mixed_corpus/README.md's discovery-only
    error-analysis discipline: don't relax scoring just because a number
    looks bad without first checking WHY), but on this corpus's descriptive
    fact_rx_indication/fact_figure_caption cases it produced a 0% and 3%
    score respectively on answers a human reading them would call correct —
    e.g. expected "COMBOGESIC is indicated in adults for the short-term
    management of mild to moderate acute pain." vs. the actual answer
    "COMBOGESIC is indicated for the short-term management of mild to
    moderate acute pain in adults." (same fact, clause order swapped).
    Only called for the minority of cases substring_ok already rejected, so
    this adds one LLM call per failure, not per case."""
    candidates = expected if isinstance(expected, list) else [expected]
    prompt = _JUDGE_PROMPT.format(question=question, expected=candidates[0], answer=answer)
    try:
        resp = await client.post(
            f"{ollama_url}/api/chat",
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "stream": False, "think": False, "options": {"temperature": 0.0, "num_predict": 5}},
            timeout=30.0,
        )
        resp.raise_for_status()
        verdict = resp.json()["message"]["content"].strip().lower()
        return verdict.startswith("yes")
    except Exception:
        return False  # judge unavailable/errored — don't silently upgrade an unverified case to "correct"


async def run_case(client, base_url, headers, case, top_k, doc_id_by_filename, judge_client=None, judge_url=None, judge_model=None):
    payload = {"question": case["question"], "top_k": top_k}
    # "scoped" cases (see eval/mixed_corpus/build_golden_dataset.py's
    # comparison_scoped type) pass document_ids the same way the frontend's
    # actual "Compare Documents" button does (api/schemas.py's QueryRequest.
    # document_ids) — a plain natural-language comparison question that
    # only NAMES two documents never reaches _augment_compare_queries()'s
    # per-document deterministic sub-query boost at all, since that's keyed
    # off document_ids being explicitly passed, not off parsing the question
    # text. Testing only the unscoped path and calling it "the comparison
    # feature" would silently never exercise the UI's actual guarantee.
    if case.get("scoped") and case.get("expected_doc_filenames"):
        doc_ids = [doc_id_by_filename[f] for f in case["expected_doc_filenames"] if f in doc_id_by_filename]
        if doc_ids:
            payload["document_ids"] = doc_ids

    r = await client.post(f"{base_url}/query", headers=headers, json=payload, timeout=60.0)
    r.raise_for_status()
    data = r.json()
    answer = data["answer"]
    sources = data.get("sources", [])
    debug = data.get("debug") or {}  # debug is explicitly null (not omitted) on early refusal

    top_chunks = debug.get("top_chunks", [])
    best_score = max((c.get("score", 0) for c in top_chunks), default=debug.get("best_rerank_score", 0))

    answered = not is_refusal(answer)

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
        # sources is the API's complete returned-source array, which can be
        # longer than top_k (identity/opening-chunk promotion can force-add
        # a document past the requested cutoff — see rag/retriever.py::
        # promote_identity_matches / promote_document_opening_chunks). Rank
        # is still computed against the full array (MRR should reflect the
        # true position), but recall_hit below is gated on rank <= top_k so
        # "Recall@5" only ever counts a hit actually within the first 5.
        doc_ids_returned = [s["document"] for s in sources]  # already ranked by relevance_score desc
        result["n_sources"] = len(doc_ids_returned)

        # expected_doc_filenames (plural) is for comparison-type questions
        # that only have a defensible answer if BOTH named documents are
        # actually retrieved. _augment_compare_queries()'s per-document
        # deterministic sub-query only fires when document_ids is actually
        # passed on the request (api/schemas.py's QueryRequest.document_ids
        # — see the "scoped" branch in run_case above) — an unscoped case
        # (question names two documents, nothing structured) exercises
        # retrieval/reranking honestly finding both from the question text
        # alone, not that boost. Score both, never conflate them: a scoped
        # miss is the "Compare Documents" UI feature failing; an unscoped
        # miss is expected/normal unless the question is unusually specific.
        expected_fnames = case.get("expected_doc_filenames") or ([expected_fname] if expected_fname else [])
        if expected_fnames:
            ranks = []
            for fname in expected_fnames:
                doc_id = doc_id_by_filename.get(fname)
                ranks.append(doc_ids_returned.index(doc_id) + 1 if doc_id and doc_id in doc_ids_returned else None)
            if all(r is not None for r in ranks):
                worst_rank = max(ranks)
                result["recall_hit"] = worst_rank <= top_k
                result["rank"] = worst_rank
                result["reciprocal_rank"] = 1.0 / worst_rank
            else:
                result["recall_hit"] = False
                result["rank"] = None
                result["reciprocal_rank"] = 0.0

        # Evidence-page recall: document-level recall_hit above only proves
        # the right DOCUMENT made it into the returned sources — it says
        # nothing about whether the specific page the fact actually lives on
        # was among the chunks the LLM was actually given. A long document
        # can satisfy recall_hit on a page-1 chunk while the page carrying
        # the queried fact (case["source_page"], set by eval/mixed_corpus/
        # build_golden_dataset.py for fact_figure_caption) never reaches the
        # prompt at all — exactly the gap that made the raw 3% substring
        # score on that type look like a flat "system is bad at this"
        # number instead of "the document was found, the chunk wasn't."
        source_page = case.get("source_page")
        if source_page is not None:
            expected_doc_id = doc_id_by_filename.get(expected_fname) if expected_fname else None
            hit_pages = [c.get("page_num") for c in top_chunks if c.get("document_id") == expected_doc_id]
            result["evidence_page_hit"] = source_page in hit_pages
            result["evidence_pages_seen"] = hit_pages

            # Evidence-CHUNK recall: evidence_page_hit above only proves
            # SOME chunk from the right page reached the model — a page
            # with 5-8 chunks routinely has the caption in just one of
            # them, so "page hit" alone conflates a real localization
            # success with a same-page-wrong-chunk miss (confirmed by
            # hand: manually re-checked 26 "page hit, answer wrong" cases
            # against real chunk text pulled from Qdrant — 14 of 26 never
            # actually had the caption text in any shown chunk, despite
            # evidence_page_hit=True for all 26). evidence_char_start/end
            # (see eval/mixed_corpus/build_golden_dataset.py) and each
            # chunk's own char_start/char_end (api/main.py's /query debug
            # output) are both offsets into the same normalize_whitespace()
            # page text, so this is a pure range comparison — no chunk text
            # needs to be sent back in debug at all.
            evidence_char_start = case.get("evidence_char_start")
            evidence_char_end = case.get("evidence_char_end")
            if evidence_char_start is not None and evidence_char_end is not None:
                same_page_chunks = [
                    c for c in top_chunks
                    if c.get("document_id") == expected_doc_id and c.get("page_num") == source_page
                    and c.get("char_start") is not None and c.get("char_end") is not None
                ]
                result["evidence_chunk_hit"] = any(
                    evidence_chunk_overlaps(c["char_start"], c["char_end"], evidence_char_start, evidence_char_end)
                    for c in same_page_chunks
                )

        expected_substr = case.get("expected_substring")
        if expected_substr and answered:
            result["substring_correct"] = substring_ok(expected_substr, normalize(answer))
            if not result["substring_correct"] and judge_client is not None:
                result["semantic_correct"] = await judge_semantic_match(
                    judge_client, judge_url, judge_model, case["question"], expected_substr, answer
                )
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
    ap.add_argument("--dataset", default=str(EVAL_DIR / "golden_dataset.json"),
                     help="Path to a dataset JSON built by build_golden_dataset.py or build_heldout_dataset.py")
    ap.add_argument("--semantic-judge", action="store_true",
                     help="On a substring_ok() miss, ask the local LLM (--judge-model) whether the "
                          "answer conveys the fact anyway before counting it wrong — see "
                          "judge_semantic_match()'s docstring for why this exists")
    ap.add_argument("--judge-url", default=os.getenv("OLLAMA_URL", "http://localhost:11435"))
    ap.add_argument("--judge-model", default=os.getenv("LLM_MODEL", "qwen2.5:7b"))
    ap.add_argument("--output", default=None,
                     help="Where to write per-case scores. Defaults next to --dataset itself "
                          "(eval/mixed_corpus/golden_dataset.json -> eval/mixed_corpus/last_run_scores_"
                          "golden_dataset.json), NOT eval/'s own — multiple corpora each have their own "
                          "golden_dataset.json, and keying purely off the stem used to silently let one "
                          "corpus's run clobber another's saved scores (caught when a mixed-corpus run "
                          "overwrote eval/last_run_scores_golden_dataset.json, the original ~400-doc "
                          "legal corpus's file, gitignored so nothing but a re-run recovers it).")
    ap.add_argument("--limit-per-type", type=int, default=None,
                     help="For fast iteration only: keep just the first N cases of each `type` "
                          "(stable dataset order, not random, so reruns during one iteration session "
                          "stay comparable to each other). Each case is a sequential /query call with "
                          "2-4 serialized Ollama round trips (query expansion, generation, sometimes a "
                          "refusal retry, sometimes --semantic-judge) — wall time scales roughly with "
                          "case count, so a small N gives fast go/no-go feedback while iterating on a "
                          "fix. NEVER treat a limited run's numbers as a real result — confirm on the "
                          "full dataset before reporting or committing to anything measured this way.")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    if args.output:
        scores_out_path = Path(args.output)
    elif args.limit_per_type:
        # Deliberately NOT the same default as a full run — see --output's
        # own docstring on the exact class of bug this avoids repeating: a
        # quick smoke run must never silently clobber the canonical
        # last_run_scores file a full run (and the docs referencing it)
        # depend on.
        scores_out_path = dataset_path.parent / f"last_run_scores_{dataset_path.stem}_limited.json"
    else:
        scores_out_path = dataset_path.parent / f"last_run_scores_{dataset_path.stem}.json"

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if args.limit_per_type:
        total = len(dataset)
        per_type_count: dict[str, int] = {}
        limited = []
        for case in dataset:
            t = case["type"]
            if per_type_count.get(t, 0) < args.limit_per_type:
                limited.append(case)
                per_type_count[t] = per_type_count.get(t, 0) + 1
        dataset = limited
        print(f"--limit-per-type {args.limit_per_type}: kept {len(dataset)}/{total} cases "
              f"({dict(sorted(per_type_count.items()))})")

    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    async with httpx.AsyncClient() as client:
        docs_resp = await client.get(f"{args.api_url}/documents", headers=headers, timeout=30.0)
        doc_id_by_filename = {d["filename"]: d["doc_id"] for d in docs_resp.json()["documents"]}

    results = []
    async with httpx.AsyncClient() as client:
        judge_client = client if args.semantic_judge else None
        for i, case in enumerate(dataset):
            try:
                res = await run_case(client, args.api_url, headers, case, args.top_k, doc_id_by_filename,
                                      judge_client=judge_client, judge_url=args.judge_url, judge_model=args.judge_model)
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
    # semantic_correct only exists on cases substring_ok already rejected
    # (see run_case) — "correct_any" folds it back in as an OR, never
    # replacing the substring number outright, so a reader always sees both:
    # what a strict verbatim check found, and how much a semantic pass on
    # top of it recovered.
    semantic_recovered = sum(1 for r in checkable if r.get("semantic_correct"))
    correct_any = correct_answers + semantic_recovered

    recall_cases = [r for r in should_answer if "recall_hit" in r]
    recall_hits = sum(1 for r in recall_cases if r["recall_hit"])
    # "In sources at all" — the API's returned-source array can exceed
    # top_k (see comment in run_case above), so this is a strictly looser
    # number than Recall@top_k and must never be reported as Recall@top_k.
    in_sources_hits = sum(1 for r in recall_cases if r.get("rank") is not None)
    max_sources = max((r["n_sources"] for r in recall_cases), default=args.top_k)
    mrr = sum(r["reciprocal_rank"] for r in recall_cases) / len(recall_cases) if recall_cases else 0

    evidence_cases = [r for r in should_answer if "evidence_page_hit" in r]
    evidence_hits = sum(1 for r in evidence_cases if r["evidence_page_hit"])

    evidence_chunk_cases = [r for r in should_answer if "evidence_chunk_hit" in r]
    evidence_chunk_hits = sum(1 for r in evidence_chunk_cases if r["evidence_chunk_hit"])

    def pct(numerator, denominator):
        return f"{100*numerator/denominator:.1f}%" if denominator else "n/a"

    print(f"\n{'='*60}")
    if args.limit_per_type:
        print(f"LIMITED RUN (--limit-per-type {args.limit_per_type}) — fast-iteration numbers only, "
              f"NOT a result. Re-run without this flag before reporting or committing anything.")
    print(f"GOLDEN DATASET EVAL — {n} cases")
    print(f"{'='*60}")
    print(f"Recall@{args.top_k}: {recall_hits}/{len(recall_cases)} ({pct(recall_hits, len(recall_cases))})")
    if max_sources > args.top_k:
        print(f"  (In returned sources at all, up to {max_sources}-wide: "
              f"{in_sources_hits}/{len(recall_cases)} ({pct(in_sources_hits, len(recall_cases))}) "
              f"— NOT Recall@{args.top_k}, do not report as such)")
    if evidence_cases:
        # Document-level Recall@k above only proves the right FILE made it
        # into sources — this checks whether the specific page the fact
        # lives on was among the chunks actually handed to the LLM (see
        # run_case's evidence_page_hit comment). Expect this to run well
        # below document recall on long, content-dense documents; that gap
        # IS the finding, not a scoring bug.
        print(f"Evidence-page Recall@{args.top_k}: {evidence_hits}/{len(evidence_cases)} ({pct(evidence_hits, len(evidence_cases))})")
    if evidence_chunk_cases:
        # Evidence-page Recall above only proves SOME chunk from the right
        # page reached the model — this checks whether the specific chunk
        # covering the fact's own char range did (see evidence_chunk_
        # overlaps' docstring). Always <= Evidence-page Recall for the same
        # run; the gap between them is exactly the "right page, wrong
        # chunk" failure mode a page-only metric can't distinguish from a
        # real generation error.
        print(f"Evidence-chunk Recall@{args.top_k}: {evidence_chunk_hits}/{len(evidence_chunk_cases)} "
              f"({pct(evidence_chunk_hits, len(evidence_chunk_cases))})")
    print(f"MRR: {mrr:.3f}")
    print(f"Overall abstention accuracy: {abstention_correct}/{n} ({pct(abstention_correct, n)})")
    print(f"  - Answered when should ({len(should_answer)} cases):  {answered_when_should}/{len(should_answer)} ({pct(answered_when_should, len(should_answer))})")
    print(f"  - Refused when should ({len(should_refuse)} cases):   {refused_when_should}/{len(should_refuse)} ({pct(refused_when_should, len(should_refuse))})")
    print(f"Answer correctness (substring match, {len(checkable)} checkable cases): {correct_answers}/{len(checkable)} ({pct(correct_answers, len(checkable))})")
    if args.semantic_judge:
        print(f"  + semantic judge recovered {semantic_recovered} more paraphrased-but-correct answers "
              f"-> {correct_any}/{len(checkable)} ({pct(correct_any, len(checkable))})")

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

    wrong_answers = [r for r in checkable if not r["substring_correct"] and not r.get("semantic_correct")]
    if wrong_answers:
        print(f"\n{len(wrong_answers)} wrong/missing-fact answers"
              f"{' (substring AND semantic judge both said no)' if args.semantic_judge else ''}:")
        for w in wrong_answers[:15]:
            print(f"  {w['id']}: {w['answer'][:120]}")

    scores_out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved per-case scores to {scores_out_path}")


if __name__ == "__main__":
    asyncio.run(main())
