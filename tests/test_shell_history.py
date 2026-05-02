import os
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import gptscan

def test_get_shell_history_paths(monkeypatch, tmp_path):
    # Mock Path.home() to return our tmp_path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Create some mock history files
    bash_hist = tmp_path / ".bash_history"
    bash_hist.write_text("ls\ncd ..")

    zsh_hist = tmp_path / ".zsh_history"
    zsh_hist.write_text("ls\ncd ..")

    # .mysql_history is missing

    paths = gptscan.get_shell_history_paths()

    assert str(bash_hist) in paths
    assert str(zsh_hist) in paths
    assert len(paths) == 2

def test_get_shell_history_paths_windows(monkeypatch, tmp_path):
    # Mock sys.platform and APPDATA
    monkeypatch.setattr(gptscan, "sys", MagicMock(platform="win32"))
    monkeypatch.setenv("APPDATA", str(tmp_path))

    # Create PowerShell history file
    ps_dir = tmp_path / "Microsoft" / "Windows" / "PowerShell" / "PSReadLine"
    ps_dir.mkdir(parents=True)
    ps_hist = ps_dir / "ConsoleHost_history.txt"
    ps_hist.write_text("dir\ncd C:\\")

    # Mock Path.home() to an empty dir to avoid finding other files
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    (tmp_path / "home").mkdir()

    paths = gptscan.get_shell_history_paths()

    assert str(ps_hist) in paths

def test_scan_shell_history_click_found(monkeypatch):
    # Mock dependencies
    mock_get_paths = MagicMock(return_value=["/mock/path/.bash_history"])
    monkeypatch.setattr(gptscan, "get_shell_history_paths", mock_get_paths)

    mock_set_target = MagicMock()
    monkeypatch.setattr(gptscan, "_set_scan_target", mock_set_target)

    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, "button_click", mock_button_click)

    gptscan.scan_shell_history_click()

    mock_set_target.assert_called_once_with(["/mock/path/.bash_history"])
    mock_button_click.assert_called_once()

def test_scan_shell_history_click_not_found(monkeypatch):
    # Mock dependencies
    monkeypatch.setattr(gptscan, "get_shell_history_paths", lambda: [])

    mock_info = MagicMock()
    monkeypatch.setattr(gptscan.messagebox, "showinfo", mock_info)

    gptscan.scan_shell_history_click()

    mock_info.assert_called_once()
    assert "No common shell history files" in mock_info.call_args[0][1]
