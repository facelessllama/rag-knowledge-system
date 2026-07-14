"""
Builds eval/eu_id_free_dataset.json — the ID-free counterpart to
eu_golden_dataset.json / eu_heldout_dataset.json (see eval/README.md,
"Methodology caveat — read this before the verdict below").

Every positive question in eu_golden_dataset.json embeds the target
document's literal CELEX ID (build_eu_golden_dataset.py:120-155), which
fires rag/retriever.py::promote_identity_matches() — a hard override that
force-includes the matching chunk above the abstention threshold
regardless of what the reranker thinks of it. That means Recall@5 and
abstention accuracy on that dataset mostly measure "does the structured
CELEX-ID index still work at scale," not "does dense+sparse semantic
search stay accurate at scale."

This dataset never puts a CELEX-ID-shaped token (see
rag/retriever.py::_CELEX_ID_QUERY_RE) in question text, so
promote_identity_matches() never fires — every question here has to be
answered by embedding/BM25/reranker quality alone, the path a real user
who doesn't know the CELEX ID would actually rely on. Three types:

- natural_lookup: refers to a document by its natural regulation number
  ("Commission Regulation (EC) No 780/2007" — note the slash, not the
  bare CELEX_ID token, and no "N of YYYY" phrasing either since that
  shape would trip retriever.py's *other* shortcut, _CASE_NUMBER_RE) or
  by a light paraphrase of its subject. Drawn from documents whose
  (normalized) subject is otherwise unique in the 57,000-doc corpus, so
  there is exactly one right answer.
- semantic_fact: same underlying facts as eu_golden_dataset.json's
  date_fact/cross_reference_fact types (reuses extract_facts() from
  build_eu_golden_dataset.py), but the document is referred to by its
  subject/number instead of its CELEX ID.
- hard_negative: the document is deliberately drawn from one of this
  corpus's large near-duplicate clusters (e.g. "establishing the standard
  import values for determining the entry price of certain fruit and
  vegetables" — 2,300+ near-identical entries here). The question
  includes the one differentiator available (adoption date) needed to
  pick a single correct document out of the cluster — directly testing
  the disambiguation-under-near-duplicate-pressure mechanism that the
  scale ladder's declining MRR pointed at, without any ID shortcut to
  fall back on.

Usage:
    python eval/build_eu_id_free_dataset.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

from build_eu_golden_dataset import (
    SOURCE_DIR,
    MANIFEST_PATH,
    _FILENAME_RE,
    extract_facts,
    ADVERSARIAL_EU,
)

EVAL_DIR = Path(__file__).resolve().parent

N_NATURAL_LOOKUP = 40
N_SEMANTIC_FACT = 40
N_HARD_NEGATIVE = 30
MIN_CLUSTER_SIZE = 5  # near-duplicate family size to qualify for hard_negative

# Same shape as rag/retriever.py::_CELEX_ID_QUERY_RE and _CASE_NUMBER_RE —
# used only as a build-time safety check, so a generated question can never
# accidentally trip promote_identity_matches()'s shortcut.
_CELEX_ID_SHAPE_RE = re.compile(r'\b\d{5}[A-Z]\d{4}(?:\(\d+\))?(?!\w)')
_CASE_NUMBER_SHAPE_RE = re.compile(r'\b\d{1,6}\s+of\s+\d{4}\b')

# Pulls (number, year, date) straight out of the filename's own descriptive
# text — cheap (no PDF open needed) and reliable for the "No <N> <YYYY> of
# <date>" citation preamble EURLEX57K filenames consistently start with.
_TITLE_NUM_DATE_RE = re.compile(
    r'No\s+(\d+)\s+(\d{2,4})\s+of\s+(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})'
)

# Everything after the citation preamble — the actual subject clause —
# starting at the first verb typical of an EU legislative title.
_SUBJECT_RE = re.compile(
    r'\b(fixing|establishing|concerning|amending|laying down|determining|'
    r'implementing|repealing|adopting|approving|correcting)\b(.+)$',
    re.IGNORECASE,
)


def assert_id_free(question: str):
    assert not _CELEX_ID_SHAPE_RE.search(question), f"CELEX-ID-shaped token leaked into: {question!r}"
    assert not _CASE_NUMBER_SHAPE_RE.search(question), f"case-number-shaped 'N of YYYY' leaked into: {question!r}"


def normalize_cluster_key(subject: str) -> str:
    """Collapses a subject clause to a near-duplicate-family fingerprint —
    strip all digits (dates, batch numbers) so weekly/monthly reissues of
    the same regulation type collapse onto the same key."""
    s = re.sub(r'\d+', '#', subject.lower())
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def build_index(manifest: list[str]) -> tuple[dict, dict]:
    """Returns (clusters, doc_info) — doc_info keyed by filename, with
    celex_id/num/year/date/subject parsed straight from the filename."""
    clusters = defaultdict(list)
    doc_info = {}
    for filename in manifest:
        m = _FILENAME_RE.match(Path(filename).stem)
        if not m:
            continue
        celex_id, description = m.groups()

        nd_match = _TITLE_NUM_DATE_RE.search(description)
        if not nd_match:
            continue
        num, year, date_str = nd_match.groups()

        subj_match = _SUBJECT_RE.search(description[nd_match.end():])
        if not subj_match:
            continue
        subject = subj_match.group(0).strip().rstrip('.')
        if len(subject) < 15:  # too short to be a meaningful, disambiguating clause
            continue

        key = normalize_cluster_key(subject)
        doc_info[filename] = {
            "celex_id": celex_id,
            "num": num,
            "year": year,
            "date_str": date_str,
            "subject": subject,
            "cluster_key": key,
        }
        clusters[key].append(filename)

    return clusters, doc_info


def build_natural_lookup(unique_docs: list[str], doc_info: dict, id_prefix: str) -> list[dict]:
    cases = []
    for i, filename in enumerate(unique_docs[:N_NATURAL_LOOKUP]):
        info = doc_info[filename]
        facts = extract_facts(SOURCE_DIR / filename)
        by_number = (i % 2 == 0)
        if by_number:
            question = f"What does {facts['type'] or 'the regulation'} (EC) No {info['num']}/{info['year']} concern?"
            qtype = "natural_lookup_by_number"
        else:
            question = f"What EU document deals with {info['subject'].lower()}, issued around {info['date_str'].split()[-1]}?"
            qtype = "natural_lookup_by_subject"
        assert_id_free(question)
        cases.append({
            "id": f"{id_prefix}_natlookup_{info['celex_id']}",
            "question": question,
            "expect_answer": True,
            "expected_doc_filename": filename,
            "expected_substring": None,  # free-form — Recall@K only
            "type": qtype,
        })
    return cases


def build_semantic_fact(unique_docs: list[str], doc_info: dict, id_prefix: str) -> list[dict]:
    cases = []
    n = 0
    for filename in unique_docs:
        if n >= N_SEMANTIC_FACT:
            break
        info = doc_info[filename]
        facts = extract_facts(SOURCE_DIR / filename)
        if not facts["date"]:
            continue
        subject_short = info["subject"].lower()
        doc_type = facts["type"] or "document"

        if n % 2 == 0:
            question = f"When was the {doc_type.lower()} {subject_short} adopted?"
            cases.append({
                "id": f"{id_prefix}_semfact_{info['celex_id']}_date",
                "question": question,
                "expect_answer": True,
                "expected_doc_filename": filename,
                "expected_substring": facts["date"],
                "type": "semantic_date_fact",
            })
        elif facts["xref_number"]:
            question = (f"What earlier regulation, directive, or decision {facts['xref_aux']} "
                        f"the {doc_type.lower()} {subject_short} {facts['xref_verb']}?")
            cases.append({
                "id": f"{id_prefix}_semfact_{info['celex_id']}_xref",
                "question": question,
                "expect_answer": True,
                "expected_doc_filename": filename,
                "expected_substring": facts["xref_number"],
                "type": "semantic_cross_reference",
            })
        else:
            continue
        assert_id_free(cases[-1]["question"])
        n += 1
    return cases


def build_hard_negatives(clusters: dict, doc_info: dict, id_prefix: str) -> list[dict]:
    cases = []
    big_clusters = sorted(
        (items for items in clusters.values() if len(items) >= MIN_CLUSTER_SIZE),
        key=len, reverse=True,
    )
    per_cluster = 2
    for cluster_docs in big_clusters:
        if len(cases) >= N_HARD_NEGATIVE:
            break
        # dedupe by date within the cluster — need each picked member's
        # date to be unmistakably its own among cluster-mates
        seen_dates = set()
        picked = 0
        for filename in cluster_docs:
            if picked >= per_cluster or len(cases) >= N_HARD_NEGATIVE:
                break
            info = doc_info[filename]
            if info["date_str"] in seen_dates:
                continue
            seen_dates.add(info["date_str"])
            subject = info["subject"].lower()
            doc_type_guess = "regulation"  # cluster subjects are near-uniformly Commission Regulations in this corpus
            question = f"What was the {doc_type_guess} {subject}, adopted on {info['date_str']}?"
            assert_id_free(question)
            cases.append({
                "id": f"{id_prefix}_hardneg_{info['celex_id']}",
                "question": question,
                "expect_answer": True,
                "expected_doc_filename": filename,
                "expected_substring": None,  # free-form — Recall@K is the point here
                "type": "hard_negative_date",
                "cluster_size": len(cluster_docs),
            })
            picked += 1
    return cases


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    clusters, doc_info = build_index(manifest)

    unique_docs = [f for f, cl in ((f, doc_info[f]["cluster_key"]) for f in doc_info)
                   if len(clusters[cl]) == 1]
    print(f"parsed {len(doc_info)}/{len(manifest)} filenames | "
          f"{len(clusters)} distinct subject clusters | "
          f"{len(unique_docs)} singleton (unique-subject) docs | "
          f"{sum(1 for c in clusters.values() if len(c) >= MIN_CLUSTER_SIZE)} clusters >= {MIN_CLUSTER_SIZE}")

    natural_lookup = build_natural_lookup(unique_docs, doc_info, "idf")
    semantic_fact = build_semantic_fact(unique_docs[N_NATURAL_LOOKUP:], doc_info, "idf")
    hard_negative = build_hard_negatives(clusters, doc_info, "idf")

    dataset = natural_lookup + semantic_fact + hard_negative + ADVERSARIAL_EU

    out_path = EVAL_DIR / "eu_id_free_dataset.json"
    out_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False))

    print(f"natural_lookup: {len(natural_lookup)}")
    print(f"semantic_fact: {len(semantic_fact)}")
    print(f"hard_negative: {len(hard_negative)}")
    print(f"adversarial: {len(ADVERSARIAL_EU)}")
    print(f"TOTAL: {len(dataset)} cases -> {out_path}")


if __name__ == "__main__":
    main()
