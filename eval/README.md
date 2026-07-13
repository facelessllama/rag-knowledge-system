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

## Current state (as of structured case metadata at ingestion)

| | golden (calibration) | heldout |
|---|---|---|
| Recall@5 | 100.0% | 100.0% |
| MRR | 1.000 | 1.000 |
| Answered when should | 100.0% (122/122) | 98.5% (67/68) |
| Refused when should | 100% (15/15) | 100% (15/15) |
| Answer correctness (substring) | 97.6% (80/82) | 100% (29/29) |

(Answer correctness moves a point run-to-run — `date_EN_0084`/`_0088` are
generation-level substring misses on the LLM's exact wording, at temperature
0.1 but not 0, unrelated to retrieval/threshold. The one heldout abstention
miss is `ho_case_lookup_by_number_A.B.A._493_2020_RAJKUMAR_YADAV_vs_THE_ST` —
see "New, unrelated observation" below; retrieval for it is verified perfect.)

`reranker.rerank()` now scores `chunk_context_text()` (title-prefixed) instead
of bare `chunk.text` — see "Next steps" history below, item 1. This closed the
`case_summary` gap outright: 12/20 (60%) → 20/20 (100%) on heldout. The
cross-encoder previously judged boilerplate-only text ("Heard learned counsel
for the petitioner...") with no way to tell documents apart; it now sees the
party names from the title directly, which is exactly what `case_summary`
questions ask about. `case_lookup_by_number` also went 18/19 → 19/19 —
`MACA_80` (previously written off as a data-completeness edge case, since its
extracted PDF text has no case number) now resolves via the title's party
names instead. One remaining failure — `date_EN_0088` (Animals Scientific
Procedures Act 1986 Fees Order 2012) — is a pre-existing, unrelated
substring-match miss on the golden set, not something this change touched.

This raised a real question — was `RELEVANCE_THRESHOLD=1.5` (calibrated
*before* this change) still valid against the new score distribution? Checked
directly: pooling both datasets (220 cases), should-refuse scores now top out
at 1.00 and should-answer scores bottom out at 4.94 — a 3.9-point gap, and
critically, this gap is uniform across every question type, not just
`case_summary`. The title-context fix lifted *all* should-answer scores
together, not only the ones that were previously borderline. That means the
per-question-type/adaptive threshold originally planned as item 2 below turned
out to be unnecessary — a single global value has ample margin now. Moved
`RELEVANCE_THRESHOLD` to `3.0` (the midpoint of the new gap, vs. the old 1.5
which was hugging the should-refuse edge) for more headroom; re-ran
`run_eval.py` on both datasets to confirm no regression (see "Current state"
table above — unaffected).

## Structured case metadata at ingestion

Item 3 from "Next steps" below: `ingestion/chunker.py::extract_case_metadata()`
now parses each case PDF's filename (`<TYPE>_<NUM>_<YEAR>_<PARTY1>_vs_
<PARTY2>.pdf`) into structured `case_number`/`case_year`/`parties` fields at
ingestion time, stored on every chunk's Qdrant payload (indexed — see
`_PAYLOAD_INDEXES` in `vector_db/qdrant_client.py`) instead of being
re-derived ad hoc by regexing each candidate chunk's own text at query time.
Two concrete upgrades this enabled in `rag/retriever.py`:

1. **Guaranteed retrievability by case number.** Previously, a case-number
   match could only be marked on a chunk that hybrid search's top-`pool`
   candidates happened to already include — if the target document scored
   too low semantically to make that pool, the exact-match promotion
   (`promote_identity_matches`) had nothing to promote. `chunks_by_case_number()`
   now does a direct Qdrant payload-filtered lookup for any (number, year)
   found in the query and injects those chunks into the candidate pool
   regardless of semantic ranking — this stops being a bet that gets worse
   as the corpus grows.
2. **Party-name matching for `case_summary`.** New `extract_party_match()`
   promotes a chunk when *all* of its document's party names appear in the
   query — as strong an identity signal as an exact case-number match, and
   available for the first time since parties weren't structured data
   before. Requires all parties (not just one) specifically because many
   filenames in this corpus share a common party (e.g. "... vs STATE OF
   ODISHA") — matching on just one would over-promote unrelated cases.

Existing corpus (400 docs, ~9800 chunks) backfilled via
`scripts/backfill_case_metadata.py` — a payload patch from each point's
already-stored `filename`, no re-parsing/re-embedding needed (4747 chunks
matched the case-file convention and got tagged; 5074 skipped, expected —
those are the plain-text UK statutory instruments with no case number).

Measured via `run_eval.py`: Recall@5 and refuse-when-should are unchanged at
100%. One new heldout abstention miss appeared
(`ho_case_lookup_by_number_A.B.A._493_2020_...`) — investigated directly
(queried the live API, inspected `debug.top_chunks`): retrieval is perfect,
all 8 reranked chunks are from the correct document at scores 8.0-8.3
(`source: "hybrid"` throughout — this case's target was already found by
ordinary hybrid search, this change's new code paths didn't even fire for
it). The LLM's generated answer starts with a refusal despite clearly
relevant context, and reproduces after a full API process restart (not
random-seed noise). This is a generation-stage issue, not a retrieval
regression from this change — flagged below as a new, separate follow-up.

## New, unrelated observation: a deterministic generation refusal

`ho_case_lookup_by_number_A.B.A._493_2020_RAJKUMAR_YADAV_vs_THE_ST` — the
model (qwen2.5:7b) answers "I could not find this information..." given
retrieved context that unambiguously contains the case caption, party
names, and case number, and reproduces the same refusal after a fresh
server restart. Worth investigating (prompt_builder's instructions,
whether the retrieved chunks lack a clear "what happened" summary for this
specific bail-order document vs. others that succeed), but out of scope for
retrieval/ingestion work — not counted as a retrieval failure above.

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

1. ~~Pass `chunk_context_text()` (see `ingestion/chunker.py`) to
   `reranker.rerank()`, not just to embedding~~ — done (`rag/reranker.py`).
   Closed the `case_summary` gap; see "Current state" above.
2. ~~Per-question-type (or otherwise calibrated/adaptive)
   `RELEVANCE_THRESHOLD` instead of one global value~~ — investigated, turned
   out unnecessary: item 1's fix lifted should-answer scores uniformly across
   types, so a single recalibrated value (`3.0`, see "Current state" above)
   has a comfortable 3.9-point margin on every type. Revisit only if a future
   corpus/question-type shows a similarly borderline gap that a global
   threshold can't cover.
3. ~~Structured case-number metadata extracted at ingestion time (not just
   matched ad-hoc against query text) would make `case_summary`-style
   lookups by party name more robust too, if paired with a metadata index~~
   — done (`ingestion/chunker.py::extract_case_metadata`,
   `vector_db/qdrant_client.py::chunks_by_case_number`,
   `rag/retriever.py::extract_party_match`). See "Structured case metadata
   at ingestion" above.
4. Investigate the deterministic generation refusal on
   `ho_case_lookup_by_number_A.B.A._493_2020_...` (see "New, unrelated
   observation" above) — retrieval is confirmed correct, so this is a
   prompt_builder/generator question, not a retrieval one.
