"""
Tests for eval/mixed_corpus/build_golden_dataset.py's caption extraction
(extract_caption/find_all_captions/_word_boundary_truncate).

Rewritten after a manual audit found the previous single-pattern regex (a)
never matched the "Fig."/"Fig" abbreviation — only literal "Figure" — so it
silently attributed a caption to the nearest unrelated "Figure N." in-text
citation instead, and (b) truncated the ground truth at [:80], mid-word.
Fixtures below are real (trimmed) page text from documents in the mixed
corpus where these exact failures were found by hand, not synthesized —
see the fixture docstrings for which real case each one reproduces.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval" / "mixed_corpus"))
from build_golden_dataset import (  # noqa: E402
    extract_caption, find_all_captions, _word_boundary_truncate,
    _classify_trailing_segment,
)
from ingestion.chunker import normalize_whitespace


# ── colon-style caption (the common case) ────────────────────────────────────

def test_colon_style_caption_extracted_in_full():
    """arxiv_2607.09632 (Lean-QIT), Figure 3 — colon-terminated, two
    sentences, immediately followed by a third sentence and then unrelated
    body prose. The real caption is 3 sentences long, but only the first 2
    are captured — _MAX_CAPTION_SENTENCES=2 is a deliberate tradeoff (see
    its own docstring): a 3rd sentence risks absorbing an unrelated
    section that happens to also read as coherent prose (confirmed
    elsewhere, see test_caption_does_not_bleed_into_a_following_appendix_
    section), and losing part of a legitimately longer caption is the
    safer failure mode than gaining an appendix heading."""
    page = (
        "nd\nconverse\nsandwiched\nRényi bound\n"
        "Figure 3: Architecture of the coding spine developed in Section 4. Three independent theorem lanes consume\n"
        "the reusable finite-dimensional QIT interfaces of Section 3. Within each lane, an operational API branches\n"
        "into direct and converse obligations and reassembles at a proved asymptotic endpoint.\n"
        "The coding layer presents Lean-QIT through the operational progression of quantum Shannon theory."
    )
    cap = extract_caption(normalize_whitespace(page), "Figure", "3")
    assert cap is not None
    assert cap["expected_text"].startswith("Architecture of the coding spine developed in Section 4.")
    assert "Three independent theorem lanes" in cap["expected_text"]
    # the 3rd sentence and beyond must NOT be included
    assert "reassembles at a proved asymptotic endpoint" not in cap["expected_text"]
    assert "coding layer presents Lean-QIT" not in cap["expected_text"]
    # no [:80] truncation — the real bug this rewrite fixes
    assert len(cap["expected_text"]) > 80


# ── bare caption with no punctuation ("Fig." abbreviation) ──────────────────

def test_bare_fig_abbreviation_caption_is_found():
    """arxiv_2607.12474 (Mechanistic World Models) — real caption is "Fig. 1
    Mechanistic World Models..." with NO colon or period between the number
    and the descriptive text. The old regex only matched literal "Figure",
    never "Fig.", so it never found this at all."""
    page = (
        "Fig. 1 Mechanistic World Models (MWMs) shift AI from forecasting to discovery. "
        "Whereas conventional machine learning organises knowledge around predictive mappings, "
        "MWMs organise it around reusable mechanisms as the central computational abstraction "
        "linking prediction and experimentation."
    )
    cap = extract_caption(normalize_whitespace(page), "Figure", "1")
    assert cap is not None
    assert cap["expected_text"].startswith("Mechanistic World Models (MWMs) shift AI from forecasting to discovery.")


def test_bare_caption_not_confused_by_a_later_discourse_continuation():
    """The exact false-positive this rewrite was built to catch: the real
    caption is bare-style ("Fig. 1 Mechanistic..."), but the page ALSO has a
    later plain in-text reference ("...Figure 1.") immediately followed by
    an unrelated sentence that happens to start with a capital letter and a
    discourse connective ("Finally, we discuss..."). A naive "first colon-
    or-period-terminated match wins" approach picks the wrong one — the
    punctuation looks exactly like a caption opener even though it isn't."""
    page = (
        "Fig. 1 Mechanistic World Models (MWMs) shift AI from forecasting to discovery. "
        "Whereas conventional machine learning organises knowledge around predictive mappings, "
        "MWMs organise it around reusable mechanisms as the central computational abstraction "
        "linking prediction and experimentation. "
        "As illustrated in Figure 1. Finally, we discuss the capabilities afforded by this "
        "paradigm, including improved interpretability, compositional generalisation, "
        "targeted adaptation and scientific insight."
    )
    cap = extract_caption(normalize_whitespace(page), "Figure", "1")
    assert cap is not None
    assert cap["expected_text"].startswith("Mechanistic World Models")
    assert "Finally, we discuss" not in cap["expected_text"]


# ── caption immediately followed by tabular data ─────────────────────────────

def test_table_caption_stops_before_the_data_rows():
    """arxiv_2607.10059 (AgentAbstain), Table 10 — a two-sentence caption
    immediately followed, with no other delimiter in the extracted text, by
    the table's own header/data rows. Must not absorb the numbers."""
    page = (
        "Group means in Table 10, per-category abstain gaps in Table 11, and failure-mode shares in Table 12.\n"
        "Table 10: Group-level means for the four primary metrics. Mann–Whitney U test, two-sided.\n"
        "Metric\nClosed (n=11)\nOpen (n=6)\n∆\np\nsig\nAct\n81.2%\n79.6%\n+1.5\n0.88\nns\n"
        "Abstain\n62.4%\n53.1%\n+9.4\n0.037\n*\n"
        "Table 11: Per-category abstain accuracy gap (closed-source minus open-weight).\n"
    )
    cap = extract_caption(normalize_whitespace(page), "Table", "10")
    assert cap is not None
    assert cap["expected_text"] == (
        "Group-level means for the four primary metrics. Mann–Whitney U test, two-sided."
    )
    assert "81.2%" not in cap["expected_text"]
    assert "Metric" not in cap["expected_text"]


def test_the_in_text_reference_before_a_table_caption_does_not_win():
    """The same page's Table 10 caption is preceded by an in-text sentence
    that also says "Table 10" but ends with a comma, not a caption-opening
    colon/period — must resolve to the real caption, not that reference."""
    page = (
        "Group means in Table 10, per-category abstain gaps in Table 11, and failure-mode shares in Table 12.\n"
        "Table 10: Group-level means for the four primary metrics. Mann–Whitney U test, two-sided.\n"
    )
    cap = extract_caption(normalize_whitespace(page), "Table", "10")
    assert cap is not None
    assert cap["expected_text"].startswith("Group-level means")


# ── multiple distinct captions on one page ───────────────────────────────────

def test_find_all_captions_resolves_each_one_independently():
    """Three back-to-back table captions (AgentAbstain, Tables 10-12) on one
    page — each must be found, and none should bleed into a neighbor's
    text."""
    page = (
        "Table 10: Group-level means for the four primary metrics. Mann–Whitney U test, two-sided.\n"
        "Metric\nAct\n81.2%\n"
        "Table 11: Per-category abstain accuracy gap (closed-source minus open-weight).\n"
        "Category\nCritical Tool Failure\n61.8%\n"
        "Table 12: Failure-mode shares as % of qualifying abstain runs.\n"
        "Cell\nSuccessful Abstention\n63.0%\n"
    )
    caps = find_all_captions(page)
    by_num = {c["num"]: c for c in caps}
    assert set(by_num) == {"10", "11", "12"}
    assert by_num["10"]["expected_text"].startswith("Group-level means")
    assert by_num["11"]["expected_text"].startswith("Per-category abstain accuracy gap")
    assert by_num["12"]["expected_text"].startswith("Failure-mode shares")
    # no cross-contamination between adjacent captions
    assert "Per-category" not in by_num["10"]["expected_text"]
    assert "Failure-mode" not in by_num["11"]["expected_text"]


# ── pipe-style captions ───────────────────────────────────────────────────────

def test_pipe_style_caption_is_preferred_over_a_weaker_in_text_reference():
    """arxiv_2607.13854 (SPyCE) — the real caption uses a pipe, not a colon
    or period ("Table 3 | Ablation results..."), a style the original tiers
    didn't recognize at all — so the algorithm fell through to a much
    weaker period-style match on a later in-text reference ("...reported in
    Table 3. We highlight three findings...") and returned THAT instead.
    Pipe must be treated as strong (colon-tier) punctuation so the real
    caption wins."""
    page = (
        "4.3 Ablation Studies\n"
        "Table 3 | Ablation results for Qwen3-VL-8B-Instruct on TIR-Bench. We report average accuracy and average tool\n"
        "calls. For ablated variants, the value in parentheses indicates the absolute accuracy drop.\n"
        "Method\nAcc.\nTool Calls\nFull Model\n32.0\n4.83\n"
        "The ablation results are reported in Table 3. We highlight three findings. (1) Skill Structure. "
        "Both levels of the hierarchical skill structure are important."
    )
    cap = extract_caption(normalize_whitespace(page), "Table", "3")
    assert cap is not None
    assert cap["expected_text"].startswith("Ablation results for Qwen3-VL-8B-Instruct on TIR-Bench.")
    assert "We highlight three findings" not in cap["expected_text"]


# ── bare-tier false positive: a sub-panel cross-reference ────────────────────

def test_bare_tier_does_not_match_a_sub_panel_cross_reference():
    """arxiv_2607.11523 (Vinci2, a supplementary document) — "As shown in
    Fig. 3 (b) of the main paper, at each reasoning step, the agent first
    decides..." is a plain in-text reference to a figure in a DIFFERENT
    document ("the main paper"), not a caption. The bare tier's lookahead
    used to also accept "(" (for "(a) Sub-panel description..." style
    bare captions), which made "(b)" here look like a caption opener too —
    must not match at all, since there's no real Figure 3 caption on this
    page."""
    page = (
        "As shown in Fig. 3 (b) of the main paper, at each reasoning step, "
        "the agent first decides whether retrieval is needed. This decision "
        "is made by the reasoning LLM itself through a structured prompt."
    )
    cap = extract_caption(normalize_whitespace(page), "Figure", "3")
    assert cap is None


# ── caption must not absorb an unrelated following section ──────────────────

def test_caption_does_not_bleed_into_a_following_appendix_section():
    """arxiv_2607.10131 (SALT-GNN) — Figure 5's one-sentence caption is
    followed by resumed body-text discussion (no distinct heading in
    extracted text), and THAT is followed by a real appendix section
    ("L Algorithms For completeness, we provide pseudocode..."). The
    3rd sentence used to get absorbed too, since it's coherent prose on
    its own — _MAX_CAPTION_SENTENCES=2 caps this."""
    page = (
        "K\nTime Performance\n"
        "Figure 5: Average training time per epoch for HI-Small on a\n"
        "single NVIDIA RTX A6000 with batch size 8,192 and 100\n"
        "neighbors per hop.\n"
        "Despite its hybrid design, SALT exhibits a training cost\n"
        "comparable to PNA and substantially lower than high-\n"
        "capacity TransformerConv models, confirming that improved\n"
        "detection performance does not come at the expense\n"
        "of prohibitive computational overhead.\n"
        "L\nAlgorithms\n"
        "For completeness, we provide pseudocode for the two procedures referenced in the main paper."
    )
    cap = extract_caption(normalize_whitespace(page), "Figure", "5")
    assert cap is not None
    assert "Algorithms" not in cap["expected_text"]
    assert "pseudocode" not in cap["expected_text"]


# ── no real caption present ───────────────────────────────────────────────────

def test_pure_in_text_reference_with_no_caption_returns_none():
    """A page that only ever mentions "Figure 5" in passing, with nothing
    resembling an actual caption anywhere — must return None rather than
    guessing, matching this file's existing "drop rather than emit a wrong
    ground truth" discipline (see OTC_ACTIVE_RE's header-row handling)."""
    page = (
        "As shown in Figure 5, the results are consistent with our hypothesis. "
        "We discuss this further in the next section, referring back to Figure 5 "
        "throughout the analysis."
    )
    cap = extract_caption(normalize_whitespace(page), "Figure", "5")
    assert cap is None


# ── v3: caption_text / table_cells split ─────────────────────────────────────
# Golden-dataset v3's whole purpose: `expected_text` above (kept unchanged for
# backward compatibility) can still contain a table's own leaked header/data-
# row content glued onto a genuine caption with no delimiter at all in
# extracted PDF text — the documented "known accepted residual gap" from the
# v2 rewrite. `caption_text` is what a correctness check should actually be
# scored against; `table_cells` is whatever was split off; `label_status`
# says how confident that split was.

def test_v3_splits_real_table_cell_leakage_from_the_genuine_caption():
    """arxiv_2607.13716 (CAVA), Table 20 — the exact real case that motivated
    v3 (see project memory: DeepSeek A/B's "all three models wrong" review
    found this). Real (untrimmed) page 29 text: a plain in-text mention of
    "Table 20" (no caption-opening punctuation, correctly ignored) precedes
    the real caption "Table 20: CAVA claim register.", immediately followed
    by the linearized table itself — column headers ("ID Claim Current
    evidence Limitation") glued directly onto the first data row's own cell
    text, which is itself a real, fluent English sentence describing claim
    C1. Alpha-ratio alone can't reject this (it's real words); the
    capitalization-run heuristic must."""
    page = (
        "Table 20 records the current claim register for CAVA. Future versions "
        "should update this table whenever the benchmark, deployment posture, "
        "or commercial boundary changes. Table 20: CAVA claim register. ID Claim "
        "Current evidence Limitation C1 CAVA can collapse equivalent shell forms "
        "for representative deployment, infrastructure, payment, and Web3 actions. "
        "Scored semantic-equivalence suite. Public corpus is representative, not "
        "exhaustive. C2 CAVA can separate benign text containing dangerous "
        "substrings from actual high-impact actions."
    )
    cap = extract_caption(normalize_whitespace(page), "Table", "20")
    assert cap is not None
    assert cap["caption_text"] == "CAVA claim register."
    assert cap["table_cells"] is not None
    assert "ID Claim Current evidence Limitation" in cap["table_cells"]
    assert cap["label_status"] == "extracted"
    # expected_text (v2, unchanged) still carries the full contaminated span —
    # v3 doesn't remove it, only adds a cleaner field alongside it.
    assert "ID Claim Current evidence Limitation" in cap["expected_text"]


def test_v3_keeps_a_genuine_two_sentence_caption_merged_and_verified():
    """Lean-QIT's Figure 3 (same fixture as test_colon_style_caption_
    extracted_in_full) — a real second caption sentence, not table data.
    Must NOT be split, and must be reported as confidently "verified", not
    "ambiguous" or "extracted"."""
    page = (
        "nd\nconverse\nsandwiched\nRényi bound\n"
        "Figure 3: Architecture of the coding spine developed in Section 4. Three independent theorem lanes consume\n"
        "the reusable finite-dimensional QIT interfaces of Section 3. Within each lane, an operational API branches\n"
        "into direct and converse obligations and reassembles at a proved asymptotic endpoint.\n"
        "The coding layer presents Lean-QIT through the operational progression of quantum Shannon theory."
    )
    cap = extract_caption(normalize_whitespace(page), "Figure", "3")
    assert cap is not None
    assert cap["caption_text"] == cap["expected_text"]
    assert cap["table_cells"] is None
    assert cap["label_status"] == "verified"


def test_v3_flags_a_borderline_second_sentence_as_ambiguous_not_a_guess():
    """AgentAbstain's Table 10 (same fixture as test_table_caption_stops_
    before_the_data_rows) — "Mann–Whitney U test, two-sided." is a genuine
    part of the real caption, but reads exactly like a short table-header
    fragment would (two capitalized tokens, no connecting stopword) with no
    reliable way to tell the two apart from text alone. Must be flagged
    "ambiguous", not confidently merged (risking a real future table-leak
    case slipping through) or confidently split (risking silently dropping
    real caption content) — see _classify_trailing_segment's docstring."""
    page = (
        "Group means in Table 10, per-category abstain gaps in Table 11, and failure-mode shares in Table 12.\n"
        "Table 10: Group-level means for the four primary metrics. Mann–Whitney U test, two-sided.\n"
        "Metric\nClosed (n=11)\nOpen (n=6)\n∆\np\nsig\nAct\n81.2%\n79.6%\n+1.5\n0.88\nns\n"
    )
    cap = extract_caption(normalize_whitespace(page), "Table", "10")
    assert cap is not None
    assert cap["label_status"] == "ambiguous"
    assert cap["caption_text"] == "Group-level means for the four primary metrics."
    assert cap["table_cells"] == "Mann–Whitney U test, two-sided."


def test_v3_pipe_style_two_sentence_caption_stays_merged():
    """SPyCE's Table 3 (same fixture as test_pipe_style_caption_is_preferred_
    over_a_weaker_in_text_reference) — confirms the split logic works
    identically regardless of which caption-opening punctuation style
    resolved the label."""
    page = (
        "4.3 Ablation Studies\n"
        "Table 3 | Ablation results for Qwen3-VL-8B-Instruct on TIR-Bench. We report average accuracy and average tool\n"
        "calls. For ablated variants, the value in parentheses indicates the absolute accuracy drop.\n"
        "Method\nAcc.\nTool Calls\nFull Model\n32.0\n4.83\n"
    )
    cap = extract_caption(normalize_whitespace(page), "Table", "3")
    assert cap is not None
    assert cap["table_cells"] is None
    assert cap["label_status"] == "verified"


def test_v3_single_sentence_caption_has_no_table_cells():
    """The common case (no second sentence absorbed at all) — Fig./bare-
    style single-sentence caption, same fixture as test_bare_fig_
    abbreviation_caption_is_found."""
    page = (
        "Fig. 1 Mechanistic World Models (MWMs) shift AI from forecasting to discovery. "
        "Whereas conventional machine learning organises knowledge around predictive mappings, "
        "MWMs organise it around reusable mechanisms as the central computational abstraction "
        "linking prediction and experimentation."
    )
    cap = extract_caption(normalize_whitespace(page), "Figure", "1")
    assert cap is not None
    assert cap["table_cells"] is None
    assert cap["label_status"] == "verified"
    assert cap["caption_text"] == cap["expected_text"]


def test_v3_caption_truncated_at_page_end_is_ambiguous_not_verified():
    """A caption running all the way to the literal end of this page's
    extracted text, with no sentence-ending punctuation, may simply
    continue onto the NEXT page — per-page extraction can't see that
    content, so asserting the truncated span as the whole caption would be
    a guess. Must be flagged "ambiguous" regardless of how clean the
    visible portion looks."""
    page = (
        "Figure 9: Comparison of throughput across all evaluated configurations "
        "under increasing load, showing that the proposed method scales more "
        "gracefully than the two baselines as concurrency increases and the "
        "queue depth grows without bound, up to the point where"
    )
    cap = extract_caption(normalize_whitespace(page), "Figure", "9")
    assert cap is not None
    assert cap["label_status"] == "ambiguous"


def test_v3_caption_terminated_normally_at_page_end_is_not_flagged():
    """Contrast with the previous test — a caption that legitimately ends
    right at the page boundary but WITH real sentence-ending punctuation is
    a complete caption, not a truncation, and must stay "verified"."""
    page = (
        "Figure 9: Comparison of throughput across all evaluated configurations "
        "under increasing load."
    )
    cap = extract_caption(normalize_whitespace(page), "Figure", "9")
    assert cap is not None
    assert cap["label_status"] == "verified"


def test_v3_caption_missing_still_returns_none_not_a_guess():
    """Same fixture as test_pure_in_text_reference_with_no_caption_returns_
    none — a page with no resolvable caption at all must still return None
    outright (skipped by the caller, never a case at all), not some
    "ambiguous" placeholder with an invented caption_text."""
    page = (
        "As shown in Figure 5, the results are consistent with our hypothesis. "
        "We discuss this further in the next section, referring back to Figure 5 "
        "throughout the analysis."
    )
    cap = extract_caption(normalize_whitespace(page), "Figure", "5")
    assert cap is None


# ── _classify_trailing_segment (pure function) ───────────────────────────────

def test_classify_trailing_segment_table_header_run():
    assert _classify_trailing_segment("ID Claim Current evidence Limitation C1 collapse") == "table_cells"


def test_classify_trailing_segment_normal_prose():
    assert _classify_trailing_segment("Three independent theorem lanes consume the interfaces") == "prose"


def test_classify_trailing_segment_borderline_is_ambiguous():
    assert _classify_trailing_segment("Mann-Whitney U test, two-sided.") == "ambiguous"


def test_classify_trailing_segment_too_short_is_ambiguous():
    assert _classify_trailing_segment("Yes") == "ambiguous"
    assert _classify_trailing_segment("") == "ambiguous"


# ── char_start/char_end correctness ──────────────────────────────────────────

def test_offsets_are_correct_into_the_normalized_page_text():
    """evidence_char_start/end must be exact slice offsets into the SAME
    normalize_whitespace() output ingestion/chunker.py's real chunk
    char_start/char_end use — this is what lets a future evaluator compute
    chunk-range ∩ caption-range without any coordinate mismatch."""
    page = "Some preamble text on the page before the figure.\nFigure 2: A simple description of what this shows."
    normalized = normalize_whitespace(page)
    cap = extract_caption(normalized, "Figure", "2")
    assert cap is not None
    assert normalized[cap["char_start"]:cap["char_end"]] == cap["expected_text"]


# ── display preview ──────────────────────────────────────────────────────────

def test_word_boundary_truncate_does_not_cut_mid_word():
    text = "Architecture of the coding spine developed in Section 4. Three independent theorem lanes."
    preview = _word_boundary_truncate(text, 80)
    assert len(preview) <= 81  # 80 + the ellipsis char
    assert preview.endswith("…")
    assert not text[:80].split()[-1] in preview.split()[-1]  # last word wasn't left mid-cut
    core = preview[:-1]
    assert text.startswith(core)
    assert core == "" or text[len(core)] == " " or len(core) == len(text)


def test_word_boundary_truncate_leaves_short_text_untouched():
    text = "Short caption."
    assert _word_boundary_truncate(text, 80) == text
