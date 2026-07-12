"""
Builds eval/heldout_dataset.json — a SEPARATE eval set from
eval/golden_dataset.json, built from documents that were never touched
while calibrating RELEVANCE_THRESHOLD (see api/main.py) or debugging
retrieval/reranking.

eval/golden_dataset.json was used both to pick the threshold value AND to
report the metrics that "prove" it works — that's circular. This dataset
exists to check whether those numbers generalize to data the tuning never
saw: documents 150-200 of the same two source pools (test_corpus/ used
0-150), plus a fresh, differently-worded adversarial set (not the exact
15 strings from build_golden_dataset.py, in case the system were somehow
keying off their specific phrasing rather than genuine topical relevance).

Usage:
    python eval/build_heldout_dataset.py
    python eval/run_eval.py --dataset eval/heldout_dataset.json
"""
import json
from pathlib import Path

from build_golden_dataset import build_answerable_from_txt, build_answerable_from_pdf

REPO_ROOT = Path(__file__).resolve().parent.parent
TXT_DIR = REPO_ROOT / "test_corpus_heldout" / "text_legal_docs"
PDF_DIR = REPO_ROOT / "test_corpus_heldout" / "pdf_court_cases"
OUT_PATH = Path(__file__).resolve().parent / "heldout_dataset.json"

# Different wording/topics from build_golden_dataset.py's ADVERSARIAL list —
# same categories where useful for comparability, but not the identical
# strings, so a pass here isn't explainable by the system having somehow
# keyed off these 15 exact phrasings during development.
ADVERSARIAL = [
    {"id": "ho_adv_capital_japan", "question": "What is the capital city of Japan?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "ho_adv_ecb_rate", "question": "What is the European Central Bank's current base interest rate?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "ho_adv_recipe_pasta", "question": "What ingredients do I need to make carbonara?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "ho_adv_weather_tokyo", "question": "Will it rain in Tokyo next week?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "ho_adv_philosophy", "question": "What did Aristotle believe about virtue ethics?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "ho_adv_canada_law", "question": "What does the Canadian Charter of Rights and Freedoms guarantee?", "expect_answer": False, "type": "adjacent_wrong_jurisdiction"},
    {"id": "ho_adv_hipaa", "question": "What are the requirements for HIPAA compliance in the US healthcare system?", "expect_answer": False, "type": "adjacent_wrong_topic"},
    {"id": "ho_adv_stock", "question": "What is the current share price of Apple Inc.?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "ho_adv_nonsense2", "question": "Seventeen singing bicycles orbited the moon on a Tuesday.", "expect_answer": False, "type": "nonsense"},
    {"id": "ho_adv_math2", "question": "What is 17 multiplied by 23?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "ho_adv_medical2", "question": "What is the standard adult dosage for paracetamol?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "ho_adv_fake_regulation2", "question": "What does the Lunar Settlement Taxation Regulations 2087 provide for?", "expect_answer": False, "type": "plausible_but_fake"},
    {"id": "ho_adv_fake_case2", "question": "What was decided in Brown vs Wilson concerning trademark infringement in New York?", "expect_answer": False, "type": "plausible_but_fake"},
    {"id": "ho_adv_olympics", "question": "Which city will host the next Winter Olympics?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "ho_adv_coding", "question": "How do I reverse a linked list in JavaScript?", "expect_answer": False, "type": "out_of_domain"},
]


def prefix_ids(cases, prefix):
    for c in cases:
        c["id"] = f"{prefix}{c['id']}"
    return cases


def main():
    txt_cases = prefix_ids(build_answerable_from_txt(txt_dir=TXT_DIR), "ho_")
    pdf_cases = prefix_ids(build_answerable_from_pdf(pdf_dir=PDF_DIR, limit=40), "ho_")
    dataset = txt_cases + pdf_cases + ADVERSARIAL
    n_answerable = sum(1 for c in dataset if c["expect_answer"])
    n_refuse = sum(1 for c in dataset if not c["expect_answer"])
    print(f"Built {len(dataset)} held-out cases: {n_answerable} answerable, {n_refuse} should-refuse")
    OUT_PATH.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
