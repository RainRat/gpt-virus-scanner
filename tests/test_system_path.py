import os
from unittest.mock import MagicMock
import pytest
import gptscan
import argparse

def test_get_system_path_directories(monkeypatch, tmp_path):
    # Create mock directories
    dir1 = tmp_path / "bin1"
    dir1.mkdir()
    dir2 = tmp_path / "bin2"
    dir2.mkdir()

    # Non-existent folder
    dir3 = tmp_path / "nonexistent"

    # Mock PATH environment variable
    mock_path = f"{dir1}{os.pathsep}{dir2}{os.pathsep}{dir3}{os.pathsep}{dir1}"
    monkeypatch.setenv("PATH", mock_path)

    dirs = gptscan.get_system_path_directories()

    # Should contain dir1 and dir2, but not dir3 (non-existent) or duplicates
    assert os.path.abspath(str(dir1)) in dirs
    assert os.path.abspath(str(dir2)) in dirs
    assert os.path.abspath(str(dir3)) not in dirs
    assert len(dirs) == 2

def test_scan_system_path_click_found(monkeypatch):
    # Mock dependencies
    mock_get_dirs = MagicMock(return_value=["/mock/bin"])
    monkeypatch.setattr(gptscan, "get_system_path_directories", mock_get_dirs)

    mock_set_target = MagicMock()
    monkeypatch.setattr(gptscan, "_set_scan_target", mock_set_target)

    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, "button_click", mock_button_click)

    gptscan.scan_system_path_click()

    mock_set_target.assert_called_once_with(["/mock/bin"])
    mock_button_click.assert_called_once()

def test_scan_system_path_click_not_found(monkeypatch):
    # Mock dependencies
    monkeypatch.setattr(gptscan, "get_system_path_directories", lambda: [])

    mock_info = MagicMock()
    monkeypatch.setattr(gptscan.messagebox, "showinfo", mock_info)

    gptscan.scan_system_path_click()

    mock_info.assert_called_once()
    assert "No valid folders found" in mock_info.call_args[0][1]

def test_cli_system_path_flag(monkeypatch):
    # This tests that the --system-path flag is accepted by the parser
    # and that the logic in main would call get_system_path_directories

    mock_get_dirs = MagicMock(return_value=["/mock/bin"])
    monkeypatch.setattr(gptscan, "get_system_path_directories", mock_get_dirs)

    mock_run_cli = MagicMock(return_value=0)
    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)

    # Mock sys.argv to simulate --system-path --cli
    monkeypatch.setattr("sys.argv", ["gptscan.py", "--system-path", "--cli"])

    # Run main() but catch SystemExit if it happens
    try:
        gptscan.main()
    except SystemExit:
        pass

    mock_get_dirs.assert_called_once()
    # Check that the directories from PATH were added to scan_targets
    args, kwargs = mock_run_cli.call_args
    scan_targets = args[0]
    assert "/mock/bin" in scan_targets
