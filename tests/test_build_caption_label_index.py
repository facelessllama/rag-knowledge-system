"""Tests for eval/mixed_corpus/build_caption_label_index.py's captions_in_text
— a thin composition of two already-tested rag.retriever functions
(extract_structural_references, structural_match_tier), used to build a
corpus-wide index of which documents contain a caption-tier match for each
Figure/Table label (see analyze_caption_misses.py's collision-count use)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval" / "mixed_corpus"))
from build_caption_label_index import captions_in_text  # noqa: E402


def test_finds_a_real_caption():
    text = "Figure 7: Comparison of accuracy across five model sizes on the benchmark suite."
    assert captions_in_text(text) == {("Figure", "7")}


def test_ignores_a_plain_in_text_reference():
    text = "As shown in Figure 7, accuracy improves with model size."
    assert captions_in_text(text) == set()


def test_finds_multiple_distinct_labels_in_one_chunk():
    text = "Table 3: Ablation results. Figure 2 | Training curves for each configuration."
    assert captions_in_text(text) == {("Table", "3"), ("Figure", "2")}


def test_no_labels_at_all():
    assert captions_in_text("This paper introduces a new method for retrieval.") == set()
