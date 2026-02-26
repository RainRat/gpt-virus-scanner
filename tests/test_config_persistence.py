import json
import pytest
from unittest.mock import patch
from gptscan import Config

@pytest.fixture
def temp_settings_file(tmp_path, monkeypatch):
    """Use a temporary settings file path for testing."""
    f = tmp_path / ".gptscan_settings.json"
    monkeypatch.setattr(Config, "SETTINGS_FILE", str(f))
    return f

@pytest.fixture(autouse=True)
def reset_config_settings():
    """Reset configuration settings before and after each test."""
    orig = {
        "last_path": Config.last_path,
        "deep_scan": Config.deep_scan,
        "git_changes_only": Config.git_changes_only,
        "show_all_files": Config.show_all_files,
        "use_ai_analysis": Config.use_ai_analysis,
        "provider": Config.provider,
        "model_name": Config.model_name,
        "threshold": Config.THRESHOLD
    }

    yield

    Config.last_path = orig["last_path"]
    Config.deep_scan = orig["deep_scan"]
    Config.git_changes_only = orig["git_changes_only"]
    Config.show_all_files = orig["show_all_files"]
    Config.use_ai_analysis = orig["use_ai_analysis"]
    Config.provider = orig["provider"]
    Config.model_name = orig["model_name"]
    Config.THRESHOLD = orig["threshold"]

def test_save_settings_writes_correct_json(temp_settings_file):
    Config.last_path = "/test/path"
    Config.deep_scan = True
    Config.git_changes_only = False
    Config.show_all_files = True
    Config.use_ai_analysis = True
    Config.provider = "ollama"
    Config.model_name = "llama3"
    Config.THRESHOLD = 75

    Config.save_settings()

    assert temp_settings_file.exists()
    with open(temp_settings_file, "r") as f:
        data = json.load(f)

    assert data["last_path"] == "/test/path"
    assert data["deep_scan"] is True
    assert data["git_changes_only"] is False
    assert data["show_all_files"] is True
    assert data["use_ai_analysis"] is True
    assert data["provider"] == "ollama"
    assert data["model_name"] == "llama3"
    assert data["threshold"] == 75

def test_load_settings_updates_config(temp_settings_file):
    settings = {
        "last_path": "/loaded/path",
        "deep_scan": False,
        "git_changes_only": True,
        "show_all_files": False,
        "use_ai_analysis": True,
        "provider": "openrouter",
        "model_name": "phi-3",
        "threshold": 40
    }
    with open(temp_settings_file, "w") as f:
        json.dump(settings, f)

    Config.load_settings()

    assert Config.last_path == "/loaded/path"
    assert Config.deep_scan is False
    assert Config.git_changes_only is True
    assert Config.show_all_files is False
    assert Config.use_ai_analysis is True
    assert Config.provider == "openrouter"
    assert Config.model_name == "phi-3"
    assert Config.THRESHOLD == 40

def test_load_settings_missing_file_no_changes(temp_settings_file):
    Config.last_path = "/unchanged"

    if temp_settings_file.exists():
        temp_settings_file.unlink()

    Config.load_settings()

    assert Config.last_path == "/unchanged"

def test_load_settings_invalid_json_graceful(temp_settings_file, capsys):
    Config.last_path = "/pre-invalid"

    with open(temp_settings_file, "w") as f:
        f.write("invalid json content")

    Config.load_settings()

    assert Config.last_path == "/pre-invalid"

    captured = capsys.readouterr()
    assert "Warning: Could not load settings" in captured.err

def test_save_settings_permission_error_graceful(temp_settings_file, capsys):
    with patch("builtins.open", side_effect=PermissionError("Access denied")):
        Config.save_settings()

    captured = capsys.readouterr()
    assert "Warning: Could not save settings" in captured.err

def test_load_settings_partial_data(temp_settings_file):
    settings = {
        "last_path": "/partial",
        "threshold": 99
    }
    with open(temp_settings_file, "w") as f:
        json.dump(settings, f)

    Config.last_path = "/old"
    Config.deep_scan = True
    Config.THRESHOLD = 50

    Config.load_settings()

    assert Config.last_path == "/partial"
    assert Config.THRESHOLD == 99
    assert Config.deep_scan is True
