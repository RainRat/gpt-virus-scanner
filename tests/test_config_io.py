import os
import pytest
from unittest.mock import patch
from gptscan import Config

@pytest.fixture(autouse=True)
def reset_config_io():
    """Reset Config state before and after each test."""
    original_apikey = Config.apikey
    original_extensions_set = Config.extensions_set.copy()
    original_extensions_missing = Config.extensions_missing
    original_taskdesc = Config.taskdesc
    original_gpt_enabled = Config.GPT_ENABLED
    original_ignore_patterns = Config.ignore_patterns.copy()

    yield

    Config.apikey = original_apikey
    Config.extensions_set = original_extensions_set
    Config.extensions_missing = original_extensions_missing
    Config.taskdesc = original_taskdesc
    Config.GPT_ENABLED = original_gpt_enabled
    Config.ignore_patterns = original_ignore_patterns

def test_save_apikey_success(tmp_path, monkeypatch):
    """Test successful saving of the API key to apikey.txt."""
    monkeypatch.chdir(tmp_path)
    Config.apikey = "test-api-key"
    Config.save_apikey()

    assert os.path.exists("apikey.txt")
    with open("apikey.txt", "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "test-api-key"

def test_save_apikey_error_handling(capsys):
    """Test error handling when saving the API key fails."""
    with patch("builtins.open", side_effect=PermissionError("Access denied")):
        Config.save_apikey()

    captured = capsys.readouterr()
    assert "Warning: Could not save API key: Access denied" in captured.err

def test_save_extensions_error_handling(capsys):
    """Test error handling when saving extensions fails."""
    with patch("builtins.open", side_effect=Exception("Disk full")):
        Config.save_extensions()

    captured = capsys.readouterr()
    assert "Warning: Could not save extensions: Disk full" in captured.err

def test_initialize_env_var_fallback_openai(monkeypatch):
    """Test that initialize falls back to OPENAI_API_KEY if apikey is empty."""
    Config.apikey = ""
    Config.taskdesc = "some task"
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    # Mock load_file to avoid side effects and noise
    with patch('gptscan.load_file', return_value=[".py"]):
        Config.initialize()

    assert Config.apikey == "env-openai-key"

def test_initialize_env_var_fallback_openrouter(monkeypatch):
    """Test that initialize falls back to OPENROUTER_API_KEY if OPENAI_API_KEY is missing."""
    Config.apikey = ""
    Config.taskdesc = "some task"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-openrouter-key")

    with patch('gptscan.load_file', return_value=[".py"]):
        Config.initialize()

    assert Config.apikey == "env-openrouter-key"

def test_initialize_no_apikey_after_env_fallback(monkeypatch, capsys):
    """Test initialize when both apikey and env vars are missing."""
    Config.apikey = ""
    Config.taskdesc = "some task"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with patch('gptscan.load_file', return_value=[".py"]):
        Config.initialize()

    captured = capsys.readouterr()
    assert Config.apikey == ""
    assert Config.apikey_missing_message in captured.err
