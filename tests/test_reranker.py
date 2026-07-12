"""
Tests for rag/reranker.py — cross-encoder scoring and the low-score
language fallback (ms-marco-MiniLM-L-6-v2 is English-only; the system is
English-only end to end, so this backstop only matters for stray
non-English input the pipeline shouldn't normally see).
"""
from unittest.mock import MagicMock, patch


def _chunk(text, score):
    return {"text": text, "score": score}


def _reranker_with_mock_model():
    with patch("sentence_transformers.CrossEncoder") as MockCE:
        from rag.reranker import CrossEncoderReranker
        instance = MagicMock()
        MockCE.return_value = instance
        reranker = CrossEncoderReranker(model_name="fake-model")
        return reranker, instance


def test_rerank_uses_cross_encoder_scores_to_reorder():
    reranker, mock_model = _reranker_with_mock_model()
    mock_model.predict.return_value = [-8.0, 4.8]
    chunks = [_chunk("irrelevant passage about jurors", 1.0),
              _chunk("the maximum discount sum is 75000", 0.2)]

    result = reranker.rerank("What is the maximum discount?", chunks, top_k=2)

    mock_model.predict.assert_called_once()
    # Cross-encoder score should override the (misleading) vector score order
    assert result[0]["text"] == "the maximum discount sum is 75000"
    assert result[0]["rerank_score"] == 4.8


def test_rerank_falls_back_to_vector_score_when_all_scores_are_very_low():
    reranker, mock_model = _reranker_with_mock_model()
    mock_model.predict.return_value = [-11.0, -12.0]
    chunks = [_chunk("a", 0.4), _chunk("b", 0.9)]

    result = reranker.rerank("some query", chunks, top_k=2)

    assert [c["rerank_score"] for c in result] == [0.9, 0.4]


def test_rerank_respects_top_k():
    reranker, mock_model = _reranker_with_mock_model()
    mock_model.predict.return_value = [1.0, 2.0, 3.0]
    chunks = [_chunk("a", 0.1), _chunk("b", 0.2), _chunk("c", 0.3)]

    result = reranker.rerank("query", chunks, top_k=2)

    assert len(result) == 2
    assert result[0]["text"] == "c"


def test_rerank_returns_empty_for_no_chunks():
    reranker, mock_model = _reranker_with_mock_model()
    assert reranker.rerank("query", [], top_k=5) == []
    mock_model.predict.assert_not_called()
