"""
Tests for eval/compare_generators.py's ModelSpec parsing — the
provider/model_id[=label] CLI spec format, and its error handling
(missing model_id, unknown provider, missing DEEPSEEK_API_KEY).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import pytest
from compare_generators import ModelSpec  # noqa: E402


def test_ollama_spec_defaults_label_to_model_id():
    spec = ModelSpec("ollama/qwen2.5:7b", ollama_url="http://x", deepseek_api_key=None)
    assert spec.provider == "ollama"
    assert spec.model_id == "qwen2.5:7b"
    assert spec.label == "qwen2.5:7b"


def test_deepseek_spec_defaults_label_to_model_id():
    spec = ModelSpec("deepseek/deepseek-v4-flash", ollama_url="http://x", deepseek_api_key="sk-test")
    assert spec.provider == "deepseek"
    assert spec.model_id == "deepseek-v4-flash"
    assert spec.label == "deepseek-v4-flash"


def test_explicit_label_override():
    spec = ModelSpec("ollama/qwen2.5:7b=qwen_local", ollama_url="http://x", deepseek_api_key=None)
    assert spec.label == "qwen_local"
    assert spec.model_id == "qwen2.5:7b"  # the "=label" split must not eat part of the model_id


def test_missing_model_id_raises():
    with pytest.raises(ValueError, match="provider/model_id"):
        ModelSpec("ollama", ollama_url="http://x", deepseek_api_key=None)


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        ModelSpec("openai/gpt-4", ollama_url="http://x", deepseek_api_key=None)


def test_deepseek_without_api_key_exits_clearly():
    with pytest.raises(SystemExit, match="DEEPSEEK_API_KEY"):
        ModelSpec("deepseek/deepseek-v4-flash", ollama_url="http://x", deepseek_api_key=None)
