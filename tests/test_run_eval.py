"""
Tests for eval/run_eval.py's pure helper functions — currently just
evidence_chunk_overlaps(), the Evidence-chunk Recall building block.
Evidence-page Recall (see run_case's evidence_page_hit) only proves SOME
chunk from the right page reached the model; this checks whether the
specific chunk covering a fact's own char range did, using char offsets
already on every real chunk (api/main.py's /query debug output) and every
caption case (eval/mixed_corpus/build_golden_dataset.py's evidence_char_
start/end) — both in the same normalize_whitespace() coordinate space, so
this is a pure range comparison, no chunk text needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from run_eval import evidence_chunk_overlaps  # noqa: E402


def test_chunk_containing_the_full_caption_is_a_hit():
    assert evidence_chunk_overlaps(chunk_start=0, chunk_end=500, cap_start=100, cap_end=200)


def test_chunk_containing_only_the_captions_opening_is_a_hit():
    """Even if the chunk doesn't cover the WHOLE caption, containing its
    opening character is the single strongest signal — the model at least
    saw where the caption starts."""
    assert evidence_chunk_overlaps(chunk_start=90, chunk_end=120, cap_start=100, cap_end=400)


def test_chunk_covering_most_of_a_caption_but_not_its_start_is_a_hit():
    """A caption can legitimately span more than one chunk — a chunk
    covering the bulk of it (>= min_coverage) still counts even without
    the opening char."""
    # cap spans [100, 200) — length 100. chunk covers [150, 200) — 50/100 = 50%.
    assert evidence_chunk_overlaps(chunk_start=150, chunk_end=200, cap_start=100, cap_end=200, min_coverage=0.5)


def test_chunk_with_only_a_sliver_of_overlap_is_not_a_hit():
    """A chunk that merely brushes the caption's edge by a few characters
    (routine given chunk_overlap between adjacent chunks) must not count —
    it tells us nothing about whether the model actually saw the fact."""
    # cap spans [100, 200) — length 100. chunk covers [195, 210) — 5/100 = 5%, no opening.
    assert not evidence_chunk_overlaps(chunk_start=195, chunk_end=210, cap_start=100, cap_end=200, min_coverage=0.5)


def test_no_overlap_at_all_is_not_a_hit():
    assert not evidence_chunk_overlaps(chunk_start=0, chunk_end=50, cap_start=100, cap_end=200)


def test_chunk_immediately_adjacent_with_no_overlap_is_not_a_hit():
    """Touching boundaries (chunk_end == cap_start) share zero characters."""
    assert not evidence_chunk_overlaps(chunk_start=0, chunk_end=100, cap_start=100, cap_end=200)


def test_min_coverage_threshold_is_respected():
    # cap spans [0, 100) — chunk covers [50, 100) — exactly 50%.
    assert evidence_chunk_overlaps(chunk_start=50, chunk_end=100, cap_start=0, cap_end=100, min_coverage=0.5)
    # cap spans [0, 100) — chunk covers [60, 100) — 40%, below a 50% threshold and no opening.
    assert not evidence_chunk_overlaps(chunk_start=60, chunk_end=100, cap_start=0, cap_end=100, min_coverage=0.5)
