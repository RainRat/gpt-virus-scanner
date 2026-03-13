import pytest
from unittest.mock import MagicMock, patch
import gptscan
from pathlib import Path
from gptscan import add_to_ignore_file, remove_from_ignore_file

@pytest.fixture
def mock_gui(monkeypatch):
    mock_root = MagicMock()
    monkeypatch.setattr(gptscan, 'root', mock_root)
    # Ensure gptscan.tk points to our mock_tk from conftest/sys.modules
    import tkinter as tk
    monkeypatch.setattr(gptscan, 'tk', tk)
    return mock_root

def test_manage_exclusions_load(mock_gui, monkeypatch):
    monkeypatch.setattr(gptscan.Config, 'ignore_patterns', ['pattern1', 'pattern2'])

    # Patch the Toplevel class in the tkinter module that gptscan is using
    with patch('tkinter.Toplevel') as mock_toplevel:
        mock_win = MagicMock()
        mock_toplevel.return_value = mock_win

        gptscan.manage_exclusions()

        mock_toplevel.assert_called_once()

def test_add_pattern_logic(tmp_path, monkeypatch):
    # Setup temporary .gptscanignore
    ignore_file = tmp_path / ".gptscanignore"
    ignore_file.write_text("old_pattern\n")
    monkeypatch.chdir(tmp_path)

    # Use actual functions instead of re-implementing
    monkeypatch.setattr(gptscan.Config, 'ignore_patterns', ["old_pattern"])

    add_to_ignore_file("new_pattern")

    assert "new_pattern" in gptscan.Config.ignore_patterns
    assert "new_pattern" in ignore_file.read_text()

def test_remove_pattern_logic(tmp_path, monkeypatch):
    ignore_file = tmp_path / ".gptscanignore"
    ignore_file.write_text("pattern1\npattern2\n")
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(gptscan.Config, 'ignore_patterns', ["pattern1", "pattern2"])

    remove_from_ignore_file(["pattern1"])

    assert "pattern1" not in gptscan.Config.ignore_patterns
    assert "pattern2" in gptscan.Config.ignore_patterns
    assert "pattern1" not in ignore_file.read_text()
    assert "pattern2" in ignore_file.read_text()
