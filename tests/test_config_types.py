import json
import pytest
from gptscan import Config

@pytest.fixture
def temp_settings_file(tmp_path, monkeypatch):
    """Use a temporary settings file path for testing."""
    f = tmp_path / ".gptscan_settings_types.json"
    monkeypatch.setattr(Config, "SETTINGS_FILE", str(f))
    return f

@pytest.fixture(autouse=True)
def reset_config_settings():
    """Reset configuration settings before and after each test."""
    orig = {
        "deep_scan": Config.deep_scan,
        "git_changes_only": Config.git_changes_only,
        "show_all_files": Config.show_all_files,
        "scan_all_files": Config.scan_all_files,
        "use_ai_analysis": Config.use_ai_analysis,
        "threshold": Config.THRESHOLD
    }
    yield
    Config.deep_scan = orig["deep_scan"]
    Config.git_changes_only = orig["git_changes_only"]
    Config.show_all_files = orig["show_all_files"]
    Config.scan_all_files = orig["scan_all_files"]
    Config.use_ai_analysis = orig["use_ai_analysis"]
    Config.THRESHOLD = orig["threshold"]

def test_load_settings_bool_strings(temp_settings_file):
    settings = {
        "deep_scan": "true",
        "git_changes_only": "1",
        "show_all_files": "False",
        "scan_all_files": "0",
        "use_ai_analysis": "yes"
    }
    with open(temp_settings_file, "w") as f:
        json.dump(settings, f)

    Config.load_settings()

    assert Config.deep_scan is True
    assert Config.git_changes_only is True
    assert Config.show_all_files is False
    assert Config.scan_all_files is False
    assert Config.use_ai_analysis is True

def test_load_settings_bool_garbage_fallback(temp_settings_file):
    settings = {
        "deep_scan": "not-a-bool"
    }
    with open(temp_settings_file, "w") as f:
        json.dump(settings, f)

    Config.deep_scan = True
    Config.load_settings()
    assert Config.deep_scan is True # Should remain unchanged

    Config.deep_scan = False
    Config.load_settings()
    assert Config.deep_scan is False # Should remain unchanged

def test_load_settings_int_strings(temp_settings_file):
    settings = {
        "threshold": "85"
    }
    with open(temp_settings_file, "w") as f:
        json.dump(settings, f)

    Config.THRESHOLD = 50
    Config.load_settings()
    assert Config.THRESHOLD == 85
    assert isinstance(Config.THRESHOLD, int)

def test_load_settings_int_garbage_fallback(temp_settings_file):
    settings = {
        "threshold": "invalid"
    }
    with open(temp_settings_file, "w") as f:
        json.dump(settings, f)

    Config.THRESHOLD = 42
    Config.load_settings()
    assert Config.THRESHOLD == 42
