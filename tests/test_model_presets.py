import pytest
from gptscan import get_model_presets

def test_get_model_presets_openai():
    """Verify OpenAI presets are returned correctly."""
    presets = get_model_presets("openai")
    assert "gpt-4o" in presets
    assert "gpt-4o-mini" in presets
    assert len(presets) > 0

def test_get_model_presets_ollama():
    """Verify Ollama presets are returned correctly."""
    presets = get_model_presets("ollama")
    assert "llama3.2" in presets
    assert "mistral" in presets
    assert len(presets) > 0

def test_get_model_presets_openrouter():
    """Verify OpenRouter presets are returned correctly."""
    presets = get_model_presets("openrouter")
    assert "gpt-4o" in presets
    assert "anthropic/claude-3.5-sonnet" in presets
    assert len(presets) > 0

def test_get_model_presets_unknown():
    """Verify unknown provider returns empty list."""
    presets = get_model_presets("unknown_provider")
    assert presets == []

def test_get_model_presets_empty():
    """Verify empty provider string returns empty list."""
    presets = get_model_presets("")
    assert presets == []
