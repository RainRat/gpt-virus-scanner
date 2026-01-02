import pytest
from unittest.mock import patch
from gptscan import Config

@pytest.fixture(autouse=True)
def reset_config():
    """Reset Config state before and after each test."""
    # Save original state
    original_apikey = Config.apikey
    original_taskdesc = Config.taskdesc
    original_extensions_set = Config.extensions_set.copy()
    original_gpt_enabled = Config.GPT_ENABLED
    original_ext_missing = Config.extensions_missing

    yield

    # Restore original state
    Config.apikey = original_apikey
    Config.taskdesc = original_taskdesc
    Config.extensions_set = original_extensions_set
    Config.GPT_ENABLED = original_gpt_enabled
    Config.extensions_missing = original_ext_missing

def test_initialize_missing_api_key(capsys):
    """Test initialization when API key is missing."""
    Config.apikey = ""
    Config.taskdesc = "some task"

    # Mock load_file to return some extensions so we don't trigger that error
    with patch('gptscan.load_file', return_value=[".py"]):
        Config.initialize()

    captured = capsys.readouterr()
    assert Config.apikey_missing_message in captured.out
    # GPT should still be enabled if task is present (since local LLMs don't need API key)
    assert Config.GPT_ENABLED is True

def test_initialize_missing_task_file(capsys):
    """Test initialization when task file is missing."""
    Config.apikey = "some_key"
    Config.taskdesc = ""

    with patch('gptscan.load_file', return_value=[".py"]):
        Config.initialize()

    captured = capsys.readouterr()
    assert Config.task_missing_message in captured.out
    assert Config.GPT_ENABLED is False

def test_initialize_missing_extensions_file(capsys):
    """Test initialization when extensions file is missing."""
    Config.apikey = "key"
    Config.taskdesc = "task"

    # Mock load_file to return empty list (simulating missing file)
    with patch('gptscan.load_file', return_value=[]):
        Config.initialize()

    captured = capsys.readouterr()
    assert Config.extensions_missing_message in captured.out
    assert Config.extensions_set == set(Config.DEFAULT_EXTENSIONS)
    assert Config.extensions_missing is True

def test_initialize_all_present(capsys):
    """Test initialization when all configuration files are present."""
    Config.apikey = "key"
    Config.taskdesc = "task"
    custom_exts = [".java", ".cpp"]

    with patch('gptscan.load_file', return_value=custom_exts):
        Config.initialize()

    captured = capsys.readouterr()
    assert Config.apikey_missing_message not in captured.out
    assert Config.task_missing_message not in captured.out
    assert Config.extensions_missing_message not in captured.out

    assert Config.GPT_ENABLED is True
    assert Config.extensions_set == set(custom_exts)
    assert Config.extensions_missing is False
