import pytest
from unittest.mock import patch, MagicMock
import gptscan
from pathlib import Path

def test_get_downloads_paths_exists(monkeypatch):
    mock_home = Path("/home/testuser")
    monkeypatch.setattr("pathlib.Path.home", lambda: mock_home)

    # Mock Path.exists to return True for the Downloads folder
    original_exists = Path.exists
    def mock_exists(self):
        if str(self) == str(mock_home / "Downloads"):
            return True
        return False

    monkeypatch.setattr("pathlib.Path.exists", mock_exists)

    paths = gptscan.get_downloads_paths()
    assert len(paths) == 1
    assert paths[0] == str(mock_home / "Downloads")

def test_get_downloads_paths_not_exists(monkeypatch):
    mock_home = Path("/home/testuser")
    monkeypatch.setattr("pathlib.Path.home", lambda: mock_home)

    # Mock Path.exists to return False
    monkeypatch.setattr("pathlib.Path.exists", lambda self: False)

    paths = gptscan.get_downloads_paths()
    assert len(paths) == 0

def test_scan_downloads_click(monkeypatch):
    monkeypatch.setattr("gptscan.get_downloads_paths", lambda: ["/downloads"])

    target_paths = []
    def mock_set_target(paths):
        nonlocal target_paths
        target_paths = paths
    monkeypatch.setattr("gptscan._set_scan_target", mock_set_target)

    clicked = False
    def mock_button_click():
        nonlocal clicked
        clicked = True
    monkeypatch.setattr("gptscan.button_click", mock_button_click)

    gptscan.scan_downloads_click()

    assert "/downloads" in target_paths
    assert clicked

def test_cli_downloads_flag(monkeypatch):
    monkeypatch.setattr("gptscan.get_downloads_paths", lambda: ["/downloads"])

    cli_args = []
    def mock_run_cli(targets, *args, **kwargs):
        nonlocal cli_args
        cli_args = targets
        return 0
    monkeypatch.setattr("gptscan.run_cli", mock_run_cli)

    import sys
    test_args = ["gptscan.py", "--downloads", "--cli"]
    with patch.object(sys, 'argv', test_args):
        gptscan.main()

    assert "/downloads" in cli_args
