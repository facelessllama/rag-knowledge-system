#!/usr/bin/env python3
"""
Small, deliberately cheap accuracy A/B test: qwen2.5:7b (local, via Ollama)
vs DeepSeek (cloud) on the SAME retrieved context for a curated sample of
golden/held-out questions — isolates the generation stage only (retrieval
is already validated separately in eval/run_eval.py), so this measures
"does the cloud model answer better/worse given identical evidence", not
retrieval quality.

Not wired into api/main.py's default pipeline — this is a standalone,
explicitly-run script. See rag/generator.py::DeepSeekGenerator.

Usage:
    python eval/test_deepseek_accuracy.py [--n 12]
"""
import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embeddings.embedding_service import EmbeddingService
from vector_db.qdrant_client import VectorStore
from rag.retriever import HybridRetriever, promote_identity_matches
from rag.reranker import CrossEncoderReranker
from rag.prompt_builder import PromptBuilder
from rag.query_expander import QueryExpander
from rag.generator import LLMGenerator, DeepSeekGenerator, is_refusal

RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "3.0"))


def normalize(s: str) -> str:
    return re.sub(r"[,\s]", "", s.lower())


# IDs that failed (answered=False, expect_answer=True) in the most recent
# eval/run_eval.py runs against golden_dataset.json and heldout_dataset.json
# after all retrieval/reranking fixes to date (case-number exact match +
# contextual retrieval) — all case_summary (party-name-only) except one
# case_lookup_by_number data-completeness edge case (MACA_80, case number
# genuinely absent from that PDF's text, kept here as a negative control:
# no generator can fix a fact that isn't in the retrieved context).
KNOWN_FAILING_IDS = {
    "case_summary_Bail_App_211_2020_AMAN_SAGOTRA_vs_UNION_",
    "case_summary_CR._MISC._3024_2020_CHANDAN_KUMAR_vs_THE",
    "case_summary_CR._MISC._4749_2020_PRABHAT_KUMAR_MAHTO_",
    "case_summary_CR._MISC._5617_2020_RANJIT_RANJEET_SINGH",
    "case_summary_CR._MISC._8546_2020_PAWAN_KUMAR_vs_THE_S",
    "case_summary_CRM_4369_2020_PADMA_BHAKAT_ANR_vs_STATE_",
    "case_summary_CRM_6559_2020_SADIQUL_ISLAM_vs_State_of_",
    "ho_case_summary_ABLAPL_1743_2020_TUTUNA_SAHOO_vs_STATE_O",
    "ho_case_summary_ABLAPL_3466_2020_RAMA_CHANDRA_BISWAL_vs_",
    "ho_case_summary_ABLAPL_694_2020_SARBESWAR_SETHY_vs_STATE",
    "ho_case_summary_BLAPL_1862_2020_NIRANJAN_MOHARANA_vs_STA",
    "ho_case_summary_CRLMP_121_2020_SUMANTA_KUMAR_ROUT_vs_THE",
    "ho_case_summary_CRMC_9706_2001_BIPIN_KR._SAHOO_vs_BASANT",
    "ho_case_summary_CRM_1152_2019_SUBRATA_DEB_vs_State_of_We",
    "ho_case_summary_FAO_293_2019_ANURAG_KHOONTE_vs_UNION_OF_",
    "ho_case_lookup_by_number_MACA_80_2020_M_-_S._IFFCO_TOKIO_G.I.CO._",
}


def pick_sample(n_per_type=3):
    golden = json.loads((Path(__file__).parent / "golden_dataset.json").read_text(encoding="utf-8"))
    heldout = json.loads((Path(__file__).parent / "heldout_dataset.json").read_text(encoding="utf-8"))
    all_cases = golden + heldout
    by_type = {}
    for c in all_cases:
        by_type.setdefault(c["type"], []).append(c)
    sample = []
    for t, cases in by_type.items():
        sample.extend(cases[:n_per_type])
    return sample


def pick_failing():
    golden = json.loads((Path(__file__).parent / "golden_dataset.json").read_text(encoding="utf-8"))
    heldout = json.loads((Path(__file__).parent / "heldout_dataset.json").read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in golden + heldout}
    missing = KNOWN_FAILING_IDS - set(by_id)
    if missing:
        print(f"WARNING: {len(missing)} known-failing IDs not found in current datasets: {missing}")
    return [by_id[i] for i in KNOWN_FAILING_IDS if i in by_id]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-type", type=int, default=2)
    ap.add_argument("--failing-only", action="store_true",
                     help="Run only the known-failing case_summary/case_lookup_by_number IDs (see KNOWN_FAILING_IDS)")
    args = ap.parse_args()

    sample = pick_failing() if args.failing_only else pick_sample(n_per_type=args.n_per_type)
    print(f"Sample: {len(sample)} cases across {len(set(c['type'] for c in sample))} types\n")

    embedder = EmbeddingService(model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    vector_store = VectorStore(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    retriever = HybridRetriever(embedder, vector_store)
    reranker = CrossEncoderReranker()
    prompt_builder = PromptBuilder()
    query_expander = QueryExpander(
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11435"),
        model=os.getenv("QUERY_EXPANDER_MODEL", "qwen2.5:7b"),
    )
    local_gen = LLMGenerator(
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11435"),
        model=os.getenv("LLM_MODEL", "qwen2.5:7b"),
    )
    deepseek_gen = DeepSeekGenerator(api_key=os.getenv("DEEPSEEK_API_KEY"))

    results = []
    deepseek_total_tokens = 0

    for i, case in enumerate(sample):
        q = case["question"]
        expanded = await query_expander.expand(q)
        chunks = await retriever.retrieve_expanded(expanded, top_k=5)
        if not chunks:
            print(f"[{i+1}/{len(sample)}] {case['id']}: no chunks retrieved, skipping")
            continue

        top_chunks = reranker.rerank(q, chunks, top_k=5)
        top_chunks = promote_identity_matches(chunks, top_chunks, RELEVANCE_THRESHOLD)
        messages = prompt_builder.build(query=q, chunks=top_chunks)

        local_result = await local_gen.generate(messages)
        deepseek_result = await deepseek_gen.generate(messages)
        deepseek_total_tokens += deepseek_result["total_tokens"]

        def grade(answer):
            refused = is_refusal(answer)
            if not case["expect_answer"]:
                return "correct_refusal" if refused else "WRONG_should_refuse"
            if refused:
                return "WRONG_false_refusal"
            expected = case.get("expected_substring")
            if expected:
                return "correct" if normalize(expected) in normalize(answer) else "WRONG_wrong_fact"
            return "answered"  # not substring-checkable (case_summary) — just confirms it answered

        local_grade = grade(local_result["answer"])
        deepseek_grade = grade(deepseek_result["answer"])

        results.append({
            "id": case["id"], "type": case["type"], "question": q[:80],
            "local_grade": local_grade, "deepseek_grade": deepseek_grade,
            "deepseek_tokens": deepseek_result["total_tokens"],
            "local_answer": local_result["answer"],
            "deepseek_answer": deepseek_result["answer"],
            "expected_doc_filename": case.get("expected_doc_filename"),
            "top_chunk_filenames": [c.get("filename", "") for c in top_chunks[:3]],
        })

        print(f"[{i+1}/{len(sample)}] {case['type']:25} local={local_grade:20} deepseek={deepseek_grade:20} ({q[:60]})")
        if args.failing_only:
            print(f"    expected doc: {case.get('expected_doc_filename')}")
            print(f"    top retrieved: {[c.get('filename','')[:40] for c in top_chunks[:2]]}")
            print(f"    local:    {local_result['answer'][:200]}")
            print(f"    deepseek: {deepseek_result['answer'][:200]}")

    n_local_ok = sum(1 for r in results if r["local_grade"] in ("correct", "correct_refusal", "answered"))
    n_deepseek_ok = sum(1 for r in results if r["deepseek_grade"] in ("correct", "correct_refusal", "answered"))

    print(f"\n{'='*70}")
    print(f"local (qwen2.5:7b):  {n_local_ok}/{len(results)} correct")
    print(f"deepseek:            {n_deepseek_ok}/{len(results)} correct")
    print(f"DeepSeek tokens used: {deepseek_total_tokens}")
    print(f"{'='*70}")

    disagreements = [r for r in results if r["local_grade"] != r["deepseek_grade"]]
    if disagreements:
        print(f"\n{len(disagreements)} cases where local and DeepSeek disagreed:")
        for d in disagreements:
            print(f"  [{d['type']}] {d['id']}: local={d['local_grade']} vs deepseek={d['deepseek_grade']}")

    Path(__file__).with_name("deepseek_accuracy_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    asyncio.run(main())
