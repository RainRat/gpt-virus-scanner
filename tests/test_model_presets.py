import gptscan

def test_get_model_presets_openai():
    presets = gptscan.get_model_presets("openai")
    assert "gpt-4o" in presets
    assert "gpt-4o-mini" in presets
    assert len(presets) > 0

def test_get_model_presets_ollama():
    presets = gptscan.get_model_presets("ollama")
    assert "llama3.2" in presets
    assert "mistral" in presets
    assert len(presets) > 0

def test_get_model_presets_openrouter():
    presets = gptscan.get_model_presets("openrouter")
    assert "gpt-4o" in presets
    assert "anthropic/claude-3.5-sonnet" in presets
    assert len(presets) > 0

def test_get_model_presets_unknown():
    presets = gptscan.get_model_presets("unknown_provider")
    assert presets == []
