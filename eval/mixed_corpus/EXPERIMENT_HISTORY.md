# Mixed-corpus experiment history

This is the compact audit trail for the cross-domain PDF evaluation. It keeps
accepted fixes, rejected experiments, and measurement caveats in one place.
The detailed methodology and interpretation remain in [`README.md`](README.md).

Unless explicitly stated otherwise, accuracy results below are from the
**discovery split**, not calibration or heldout. The frozen v2 corpus contains
762 documents; 750 were indexed and 12 deliberately difficult scans were
rejected for having no extractable text. The fact benchmark contains 454
cases, including 177 `fact_figure_caption` cases.

## Fixed evaluation identity

| Item | Value |
|---|---|
| Corpus | `mixed_corpus_v2` |
| Qdrant collection | `mixed_corpus_v2` |
| Postgres database | `ragdb_mixed_corpus_v2` |
| Corpus manifest SHA-256 | `791ed00fda6e7f246f8a5b0032cba946b7c6b24f3957f054bfe6ea4cb1e69513` |
| Golden v2 SHA-256 | `868c638c30f30bb9e65acf6547b551c06216bab254c16d4084cf85202bceb14d` |
| Golden v3 SHA-256 | `da4165a9accf5447b27c60b028a33fbad3f3ee9c8999020f9676f7a9e86949d9` |
| Embedding model | `BAAI/bge-m3` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Local generator/judge | `qwen2.5:7b` |
| Chunking | 512 characters, overlap 50 |

See [`v2_run_metadata.json`](v2_run_metadata.json) for ingestion counts and
the exact frozen-instance notes.

## Chronology

| Stage | Commit or state | Result | Decision |
|---|---|---|---|
| Corpus construction and extraction probe | `dfd027e` | 762 PDFs across scientific, medical, FAA, and adversarial OCR slices; frozen 60/20/20 split | Accepted as the cross-domain stress corpus |
| NUL-byte ingestion regression | `b95fe44` | 215/597 arXiv documents were affected before sanitization; exact failures ingested after the fix | Accepted |
| Qdrant serialized-size batching | `d106962` | The two oversized documents, 1,494 and 1,806 chunks, ingested in two batches each | Accepted |
| Preliminary baseline | `1eab4cd`; drifting `mixed_corpus_test` | 447 cases; caption Doc Recall 97%, Evidence-page Recall 31%, semantic correctness 11% | Superseded: collection drifted and caption labels were truncated |
| Evidence-chunk metric | `f1591af` | Measures overlap with the labelled caption span rather than only the page | Accepted as the primary structural-retrieval metric |
| Frozen v2 baseline | `dfd027e` | 454 cases; caption Doc Recall 100%, Evidence-page 29.3%, Evidence-chunk 19.9%, semantic correctness 11% | Accepted baseline |
| Broad two-stage document-to-chunk expansion | uncommitted experiment, reverted | Caption Evidence-page 31%→24%, semantic correctness 11%→6%, answer rate 33%→29% | Rejected: same-document decoys flooded reranking |
| Literal Figure/Table structural lookup, top-3 documents | `8c089c7` | Caption Evidence-page 36.5%, Evidence-chunk 28.2%, semantic correctness 14%; no regressions | Accepted first structural baseline |
| Structural candidate window top-3→top-5 | uncommitted constant change, reverted | Evidence-page 36.5%→35.4%; Evidence-chunk 28.2%→27.1%; one miss→hit and three hit→miss | Rejected: widening did not solve selection and added distractors |
| Conditional generator A/B on identical pre-guard-fix contexts | tools `698238e`, docs `e09bf50`; contexts captured at `8c089c7` | n=47 evidence-hit cases: Qwen 40.4%, DeepSeek V4 Flash 76.6%, V4 Pro 72.3% | Conditional generation gap reproduced; no Flash-over-Pro claim |
| Full taxonomy of remaining structural misses | `072920b` | 127 reproduced: 112 displaced canonical chunks, 12 wrong occurrence/prose, 3 caption absent; 1 nondeterministic case excluded | Collision hypothesis rejected; fix the dominant invariant violation |
| Promote an already-present canonical structural chunk | `aa07106`, documented `1c2fc73` | Evidence-page 36%→92.7%; Evidence-chunk 28.2%→91.5%; semantic correctness 14%→39%; no regressions | Accepted |
| Title-free robustness diagnostic | script and initial run in `5f0f73e` | Global ambiguous query: 2.3% Evidence-chunk. Scoped realistic query before fix: 69.5%, with structural marker 0.0% | Global result is a non-goal; scoped guard was a real product bug |
| Structural lookup within explicit document scope | `5f0f73e`, documented `a3f2c45` | Scoped title-free Evidence-chunk 69.5%→95.5%, structural marker 0.0%→98.3%; 358 tests green | Accepted current retrieval checkpoint |
| Golden-dataset v3 (caption_text/table_cells split) | `5665b1d` | 181 caption cases: 149 verified, 22 ambiguous, 10 extracted; 31 changed from v2; v2 untouched; 370 tests green | Accepted — v2 kept as a named baseline, v3 is additive |
| Re-score saved DeepSeek A/B answers against v3 | `eval/mixed_corpus/rescore_against_v3.py` (raw per-row report gitignored) | qwen 41.7%→50.0%, Flash 77.1%→81.2%, Pro 72.9%→75.0%; most still-wrong cases are on unchanged (`verified`) labels | Generation gap confirmed real, not primarily a v2 label-scoring artifact |
| Post-fix DeepSeek A/B, full new evidence-hit population | `142462d` (`generator_ab_postfix_v3_*`) | 162/177 evidence-hit (113 newly rescued); verified-tier conditional correctness qwen 45.2%, Flash 71.1%, Pro 66.7% — close to the historical n=47/48 numbers | DeepSeek edge confirmed on a ~3.4x larger, cleaner-label population; Pro shows a real context-confusion failure mode; found a MORE severe golden-v3 blind spot (no-period leaked table/chart content bypasses the split entirely, ~17/148 verified cases suspiciously long) — not fixed, flagged |
| Title-free, document-scoped generation (separate report) | `generator_ab_postfix_v3_titlefreescoped_*` | 168/177 evidence-hit; verified-tier conditional correctness (automated judge) qwen 47.5%, Flash 52.5%, Pro 48.9% — DeepSeek edge shrinks from ~21-26pp to +5.0pp/+1.4pp | Initially reported as an OPEN, unresolved paradox — RESOLVED below |
| Paradox resolution: label audit → factor analysis → blind adjudication | `5dc92da` (`apply_manual_label_review.py`, `cross_run_factor_analysis.py`, `prepare_blind_adjudication.py`) | Manual label cleanup didn't move the numbers (rules out defective labels). Same evidence chunk 97.1% of the time across scenarios (rules out different populations), but DeepSeek got NET WORSE in the cleaner scoped context (Flash −26, Pro −23) while Qwen stayed flat (+4). Blind adjudication of all 63 disagreements (model identity + prior verdict both hidden): automated judge under-scores everyone, Qwen disproportionately (39.7% automated vs 76.2% manual-strict vs 60.3%/92.1% Flash) — 86/189 (45.5%) judge disagreements | **Resolved**: DeepSeek Flash retains a real +15.9pp (strict) edge scoped/title-free — smaller than title-bearing, not zero. Crosses the ≥10pp "add cloud mode" threshold for Flash; Pro stays excluded (documented behavioral issues). No factorial experiment needed. |

## Rejected-experiment record

### Broad two-stage expansion

This code was tried in the working tree and reverted before a commit, so its
exact patch is not recoverable from Git. The experiment took documents found
at stage 1, expanded the candidate pool with more chunks from those documents,
and let the reranker choose from the wider pool. It was motivated by high
document recall and low evidence-page recall. It failed because the added
chunks contained many nearby or repeated Figure/Table mentions, creating more
decoys rather than a stronger identity signal. The measured regression is
preserved above; the mechanism should not be reconstructed unless a new,
specific hypothesis requires it.

### Structural top-5 window

This changed only `STRUCTURAL_TOP_DOCS` from 3 to 5 and was reverted after the
full paired run. It produced one `miss→hit` and three `hit→miss` transitions.
The later full taxonomy showed that candidate-window misses were 0/127 in the
title-bearing golden questions; the dominant failure was canonical-chunk
promotion, not window width. The raw result remains local and its hash is
recorded below.

## Raw artifact integrity

Detailed per-case outputs are intentionally gitignored because they are
regenerable and noisy. These hashes identify the exact local files used for
the recorded aggregates. They are **not** available in a fresh clone unless
archived separately.

| Artifact | SHA-256 |
|---|---|
| `last_run_scores_golden_dataset.json` | `ee3c6df53a486e2ae182c3e070dc981256544736d028dc9868d5d81f738eaa81` |
| `last_run_scores_golden_dataset_v2.json` | `391f0b5bb0c161778b45433ce2a657c02890bdb9b5bd9c88af2f291f4545323c` |
| `last_run_scores_golden_dataset_v2_structural.json` | `35c2ba70416983d93ecdfbe8e7de9a29e44fa58c3ddec6d8a65a4ae9aad47a89` |
| `last_run_scores_golden_dataset_v2_structural_top5.json` | `2e2f9c58f14cdda2d6ebc65ceac83a15aa771dffed186943e5929049f5e99bbb` |
| `caption_miss_taxonomy.json` | `624f68eb5e24a069306b2120644ebf826b0dd943f9f9ffb267afc1f1c217a9ed` |
| `last_run_scores_golden_dataset_v2_structural_fix.json` | `f6dbe1f9e4ae14ada51a3342e305db482bbad5ddc9acf4fe21bf3cfa1a699376` |
| `last_run_scores_golden_dataset_v2_structural_fix2.json` | `8da9e60da3cbd8c9bb29e3e993ce5014465a9eef5b2a5393d25fe0fe5d8d8d98` |
| `title_free_retrieval_check.json` | `91bbf2fd444d702fd6bd890ead0cc0527fd30b2c4d8187f739cf1fb5e558ccde` |
| `title_free_retrieval_check_fixed.json` | `706b116c4ee44ac79b77277fc08a17378f60cb1a8f3f452c3d0c9d4d350a53b0` |
| `generator_ab_contexts.json` | `e410b0cac61292a475b9dbecd637dd4b1dede602c8b400bdfbd5556c300b0e23` |
| `generator_ab_results.json` | `6b51408f12a5eeddf394a8b6075d7e3e34f8b40c9aad450c889b9378619b2e38` |
| `rescore_v3_report.json` | `a9e0bd68b83f689a1991edf1a1c33231128f7d48e3ca9792763795eaf461906e` |
| `generator_ab_postfix_v3_contexts.json` | `40ed7a6ba5dea351651aadb696fb1c04ca1c2fb0ed881a4e1a033d16df888e72` |
| `generator_ab_postfix_v3_results.json` | `85d97aafb96eccdb73c3a645556b9b76257389f33f73d1001debceb0d1290345` |
| `generator_ab_postfix_v3_titlefreescoped_contexts.json` | `634302415e1a3cf9aff3450b4b9f540c5214cc3e97718bc90d9b96e02da99151` |
| `generator_ab_postfix_v3_titlefreescoped_results.json` | `53ea622ec234645af8ff11ff975b00918033a74b4e4b140bfe8e9654c8cfab3d` |
| `analyze_postfix_ab_titlefreescoped_report.json` | `b4a32fdf5276ad1d64064987540ea3ac00646410820e83ff8464e62a20a6d4ef` |
| `analyze_postfix_ab_report.json` | `210a523bec5d66516fa7ca711f2b24397c4043b11048efec85210f5c37ff58e6` |

`blind_adjudication_verdicts.json`, `blind_adjudication_titlefreescoped_
KEY.json`, and `blind_adjudication_report.json` are committed directly
(the non-regenerable human-judgment record and its small aggregate) —
not gitignored, no hash needed here.

The tracked [`generator_ab_report.md`](generator_ab_report.md) preserves the
aggregate generator results and paired contingency tables even if the raw API
answers are unavailable.

## Current interpretation

At the frozen-v2 starting point, document-level recall hid a severe
within-document localization failure: Evidence-chunk Recall was only 19.9%.
The first narrow structural lookup improved it to 28.2%. A complete miss
taxonomy then exposed a single invariant violation affecting 112 cases; fixing
it raised unscoped title-bearing retrieval to 91.5%. Testing a more realistic
workflow exposed a second guard bug, and explicit document scope now reaches
95.5% without repeating the document title.

Therefore the historical claim that retrieval is necessarily the dominant
limiter no longer applies to the current scoped-caption workflow. The old
generator A/B still proves that model choice matters once evidence is present,
but its end-to-end projection was tied to the pre-fix 28.2% retriever. Current
end-to-end model quality must be measured again after label cleanup.

Golden v3 (149 verified / 22 ambiguous / 10 extracted, 31 cases changed from
v2) closes the label-noise question specifically: re-scoring the already-saved
DeepSeek A/B answers against it moved absolute correctness only modestly
(qwen +8.3pp, Flash +4.1pp, Pro +2.1pp), and the large majority of remaining
wrong answers sit on labels v3 never touched (`verified`). The conditional
generation gap this pass measured is therefore real, not primarily an
artifact of the v2 table-cell-leak defect.

The post-fix full-population A/B (162/177 evidence-hit, 113 newly rescued)
confirms this at scale: verified-tier conditional correctness (qwen 45.2%,
Flash 71.1%, Pro 66.7%) sits close to the historical n=47/48 numbers
(40.4/76.6/72.3%) despite a ~3.4x larger and cleaner-labeled population —
the DeepSeek edge is not a fluke of the original small sample. Manual review
of every disagreement surfaced two real findings: Pro has a specific
context-attribution-confusion/truncation failure mode Flash does not share,
and golden v3 has a MORE severe blind spot than the one it was built to fix
— leaked table/chart content with no period anywhere in it never reaches
`_classify_trailing_segment` at all, since the split only fires on a second
sentence-splitter piece. All 17 were hand-audited (`apply_manual_label_
review.py`): 7 genuine (`verified_reviewed`), 10 real defects incl. one
outright misattribution (`ambiguous`) — see below, this did NOT explain
the title-free-scoped paradox.

**The title-free-scoped paradox (DeepSeek's edge appearing to nearly
vanish without the title) is RESOLVED**, via a three-stage decomposition
with no new generator API calls until the final stage: (1) the manual
label audit above ruled out defective labels — the numbers barely moved.
(2) `cross_run_factor_analysis.py` (local retrieval only) found the same
evidence chunk 97.1% of the time across both scenarios (rules out
comparing different populations) but DeepSeek getting NET WORSE in the
cleaner, single-document, rank-1 scoped context (Flash −26, Pro −23
correct→wrong minus wrong→correct) while Qwen stayed flat (+4) — the
opposite of "cleaner context helps the weaker model." (3) Blind manual
adjudication of all 63 disagreement cases (model identity AND the prior
automated verdict both hidden) found the automated judge under-scores
every model but Qwen disproportionately (39.7% automated vs. 76.2%
manual-strict; Flash 60.3% vs. 92.1%) — 86/189 (45.5%) judge
disagreements. **Resolved conclusion**: DeepSeek Flash retains a real
+15.9pp (strict) edge scoped/title-free — smaller than the title-bearing
A/B's ~21-26pp, but genuinely non-zero, crossing the user's own ≥10pp
"add optional cloud mode" threshold. Pro stays excluded regardless
(documented behavioral issues). See `eval/mixed_corpus/README.md`'s
"Separate check" section for full detail.

## Open work, in order

1. ~~Build golden dataset v3, separating `caption_text` from adjacent
   table-cell content while preserving v2 unchanged~~ — done (`5665b1d`).
2. ~~Rescore the already-saved Qwen/DeepSeek answers against v3 labels
   without making new API calls~~ — done (`eval/mixed_corpus/rescore_
   against_v3.py`); generation gap confirmed real, see above.
3. ~~Capture contexts from the current post-fix retriever and repeat the
   conditional generator comparison~~ — done (`142462d`,
   `generator_ab_postfix_v3_*`); DeepSeek edge confirmed at scale, Pro
   failure mode found, new golden-v3 blind spot found (see above).
   ~~Separate title-free-scoped generation check~~ — also done
   (`generator_ab_postfix_v3_titlefreescoped_*`), and its apparent
   near-vanishing DeepSeek edge ~~is an OPEN, unresolved finding~~ was
   RESOLVED (`5dc92da`) via blind adjudication: Flash retains a real
   +15.9pp (strict) edge — see above.
4. Golden v4: detect table/chart-like content leaking into a caption's
   FIRST sentence-splitter piece (no second piece required to trigger),
   not just the second-piece case v3 already handles. Motivated by the
   ~17/148 suspiciously-long `verified` cases found above — NOT started.
5. ~~Distinguish the title-free-scoped narrowing from scorer noise~~ —
   done via blind adjudication (`5dc92da`), see above. Judge behavior
   with title-free question text specifically may be worth revisiting
   generally (not just for this comparison) — NOT started, low priority.
6. **Decide on the generator**: per the resolved numbers and the user's
   own decision rule, adding DeepSeek Flash as an optional cloud mode is
   now justified (edge ≥10pp, no documented behavioral issues unlike
   Pro). Not yet implemented in the product — a product decision, not an
   eval-pass task.
7. Use calibration for final thresholds/configuration.
8. Run heldout once after the approach is frozen; do not tune on it.
9. Close the historical `comparison_unscoped` empty-exception case as either
   root-caused or explicitly non-reproducible.

The remaining 3 caption-absent and 12 wrong-occurrence retrieval misses are
documented residuals, not current optimization targets. The globally unscoped
question “What does Figure 7 show?” is genuinely ambiguous across hundreds of
documents and is also not a retrieval target; the product should request or
carry document scope.
