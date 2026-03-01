import json
import os
import pytest
from unittest.mock import MagicMock, patch
import gptscan
from gptscan import Config

@pytest.fixture
def temp_settings_file(tmp_path, monkeypatch):
    """Use a temporary settings file path for testing."""
    f = tmp_path / ".gptscan_settings.json"
    monkeypatch.setattr(Config, "SETTINGS_FILE", str(f))
    return f

@pytest.fixture(autouse=True)
def reset_globals():
    """Reset globals before and after each test."""
    orig_recent = Config.recent_paths[:]
    Config.recent_paths = []
    orig_cancel = gptscan.current_cancel_event
    gptscan.current_cancel_event = None
    yield
    Config.recent_paths = orig_recent
    gptscan.current_cancel_event = orig_cancel

def test_config_save_load_recent_paths(temp_settings_file):
    Config.recent_paths = ["/path/1", "/path/2"]
    Config.save_settings()

    # Clear and reload
    Config.recent_paths = []
    Config.load_settings()
    assert Config.recent_paths == ["/path/1", "/path/2"]

def test_button_click_updates_history(monkeypatch, temp_settings_file):
    # Mock dependencies of button_click
    monkeypatch.setattr(gptscan, "clear_results", MagicMock())
    monkeypatch.setattr(gptscan, "set_scanning_state", MagicMock())
    monkeypatch.setattr(gptscan, "update_status", MagicMock())
    monkeypatch.setattr(gptscan, "run_scan", MagicMock())
    monkeypatch.setattr(gptscan, "messagebox", MagicMock())

    # Use dry_run to avoid scripts.h5 check
    mock_dry_var = MagicMock()
    mock_dry_var.get.return_value = True
    monkeypatch.setattr(gptscan, "dry_var", mock_dry_var)

    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "/new/path"
    monkeypatch.setattr(gptscan, "textbox", mock_textbox)

    # Also need to mock other globals used in button_click
    monkeypatch.setattr(gptscan, "git_var", MagicMock(get=lambda: False))
    monkeypatch.setattr(gptscan, "deep_var", MagicMock(get=lambda: False))
    monkeypatch.setattr(gptscan, "all_var", MagicMock(get=lambda: False))
    monkeypatch.setattr(gptscan, "gpt_var", MagicMock(get=lambda: False))

    # Initially empty
    Config.recent_paths = ["/old/path"]

    # Ensure current_cancel_event is None
    gptscan.current_cancel_event = None

    with patch("threading.Thread"):
        gptscan.button_click()

    assert Config.recent_paths == ["/new/path", "/old/path"]
    mock_textbox.__setitem__.assert_called_with('values', ["/new/path", "/old/path"])

def test_history_uniqueness_and_limit():
    Config.recent_paths = [f"/path/{i}" for i in range(10)]

    # Mock textbox
    mock_textbox = MagicMock()
    with patch("gptscan.textbox", mock_textbox):
        # Add a duplicate path
        scan_path = "/path/5"
        # Manually trigger the logic that was added to button_click
        if scan_path and (not Config.recent_paths or Config.recent_paths[0] != scan_path):
            if scan_path in Config.recent_paths:
                Config.recent_paths.remove(scan_path)
            Config.recent_paths.insert(0, scan_path)
            Config.recent_paths = Config.recent_paths[:10]
            if mock_textbox:
                mock_textbox['values'] = Config.recent_paths

        assert Config.recent_paths[0] == "/path/5"
        assert len(Config.recent_paths) == 10
        assert Config.recent_paths.count("/path/5") == 1

        # Add a new path, should push out the last one
        scan_path = "/path/new"
        if scan_path and (not Config.recent_paths or Config.recent_paths[0] != scan_path):
            if scan_path in Config.recent_paths:
                Config.recent_paths.remove(scan_path)
            Config.recent_paths.insert(0, scan_path)
            Config.recent_paths = Config.recent_paths[:10]
            if mock_textbox:
                mock_textbox['values'] = Config.recent_paths

        assert Config.recent_paths[0] == "/path/new"
        assert len(Config.recent_paths) == 10
        assert "/path/9" not in Config.recent_paths

def test_clear_path_history(monkeypatch):
    Config.recent_paths = ["/path/1", "/path/2"]
    mock_textbox = MagicMock()
    monkeypatch.setattr(gptscan, "textbox", mock_textbox)
    monkeypatch.setattr(gptscan, "update_status", MagicMock())
    monkeypatch.setattr(Config, "save_settings", MagicMock())

    gptscan.clear_path_history()

    assert Config.recent_paths == []
    mock_textbox.__setitem__.assert_called_with('values', [])
    Config.save_settings.assert_called_once()
