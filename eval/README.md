# eval/

Regression harness for retrieval and answer quality. Not part of `pytest`
(needs a running instance with `test_corpus/` + `test_corpus_heldout/`
ingested) — run manually, see commands below.

## Files

- `build_golden_dataset.py` → `golden_dataset.json` (137 cases) — the
  calibration set. `RELEVANCE_THRESHOLD` (see `api/main.py`) was tuned
  against this set.
- `build_heldout_dataset.py` → `heldout_dataset.json` (83 cases) — documents
  and adversarial questions never touched during calibration or debugging.
  Exists because `golden_dataset.json` was used both to pick the threshold
  *and* to report the metrics claiming it works — circular. Run against
  this set to check a fix actually generalizes, not just fits what it was
  tuned on.
- `run_eval.py --dataset <path>` — runs either dataset against a live
  instance, reports Recall@K, MRR, abstention accuracy, answer correctness.
  Writes `last_run_scores_<name>.json` (gitignored).
- `test_deepseek_accuracy.py` — standalone A/B test, local (Ollama) vs
  DeepSeek (cloud) generation on identical retrieved context. Not wired
  into `api/main.py`'s default pipeline. See "DeepSeek A/B" below.

Both datasets have three question types generated per PDF filename
(`<TYPE>_<NUM>_<YEAR>_<parties>.pdf`):
- `case_summary` — party names only ("What is the case X vs Y about?")
- `case_lookup_by_number` — includes the case number ("What is case CRM
  10691 of 2020 (X vs Y) about?")
- `fact_date` / `fact_money` — programmatically extracted from
  `test_corpus/text_legal_docs/` ("come into force on <date>", single-amount
  "£<N>" clauses), plus 15 hand-written adversarial (should-refuse) questions.

## Commands

```bash
python eval/build_golden_dataset.py
python eval/build_heldout_dataset.py
python eval/run_eval.py --dataset eval/golden_dataset.json
python eval/run_eval.py --dataset eval/heldout_dataset.json
python eval/test_deepseek_accuracy.py --failing-only   # needs DEEPSEEK_API_KEY in .env
```

## Current state (as of the case-number + contextual-retrieval fixes)

| | golden (calibration) | heldout |
|---|---|---|
| Recall@5 | 95.1% | 86.8% |
| MRR | 0.943 | 0.860 |
| Answered when should | 94.3% (115/122) | 86.8% (59/68) |
| Refused when should | 100% (15/15) | 100% (15/15) |
| Answer correctness (substring) | 98.8% (81/82) | 100% (29/29) |

Remaining failures are entirely `case_summary` (party-name-only questions),
plus one `case_lookup_by_number` data-completeness edge case (`MACA_80` —
the case number is genuinely absent from that PDF's extracted text, not a
retrieval bug).

## DeepSeek A/B — the bottleneck is retrieval, not generation

Ran `test_deepseek_accuracy.py --failing-only` (16 known-failing cases,
identical retrieved context fed to both models): **qwen2.5:7b (local) and
DeepSeek (cloud) agreed on every single case — 9/16 correct, same 7
failures, zero disagreements.**

Inspecting the failures: in all 7, the expected document never appears
among the top retrieved chunks at all (completely unrelated cases surface
instead). In all 9 successes, the expected document is the top retrieval
hit. Success/failure correlates perfectly with whether retrieval found the
right document — not with which LLM generated the answer.

**Conclusion: no LLM upgrade (cloud or otherwise) will fix this class of
failure.** The fix has to be in retrieval/reranking, not generation. Don't
spend budget on a bigger/better generation model for this problem.

## Next steps (not done yet)

1. Pass `chunk_context_text()` (see `ingestion/chunker.py`) to
   `reranker.rerank()`, not just to embedding — the cross-encoder currently
   judges bare `chunk.text`, missing the same document-identity context
   that already helped the embedding/retrieval stage.
2. Per-question-type (or otherwise calibrated/adaptive) `RELEVANCE_THRESHOLD`
   instead of one global value — `case_summary`'s correct-answer scores
   cluster much lower than `fact_date`/`fact_money` even when genuinely
   correct (confirmed directly: one case ranked its correct document #1
   post-rerank at score 1.382, still short of the 1.5 gate).
3. Structured case-number metadata extracted at ingestion time (not just
   matched ad-hoc against query text, as `extract_case_numbers` in
   `rag/retriever.py` does now) would make `case_summary`-style lookups by
   party name more robust too, if paired with a metadata index.
