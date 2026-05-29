import pytest
from unittest.mock import patch, MagicMock
import gptscan
import sys
import argparse

def test_scan_ssh_config_click_success(monkeypatch):
    # Mock dependencies
    mock_get_paths = MagicMock(return_value=["/fake/ssh_config"])
    mock_set_target = MagicMock()
    mock_button_click = MagicMock()

    monkeypatch.setattr("gptscan.get_ssh_config_paths", mock_get_paths)
    monkeypatch.setattr("gptscan._set_scan_target", mock_set_target)
    monkeypatch.setattr("gptscan.button_click", mock_button_click)

    gptscan.scan_ssh_config_click()

    mock_get_paths.assert_called_once()
    mock_set_target.assert_called_once_with(["/fake/ssh_config"])
    mock_button_click.assert_called_once()

def test_scan_ssh_config_click_no_files(monkeypatch):
    # Mock dependencies
    mock_get_paths = MagicMock(return_value=[])
    mock_showinfo = MagicMock()

    monkeypatch.setattr("gptscan.get_ssh_config_paths", mock_get_paths)
    # messagebox is imported as from tkinter import messagebox
    monkeypatch.setattr("gptscan.messagebox.showinfo", mock_showinfo)

    gptscan.scan_ssh_config_click()

    mock_get_paths.assert_called_once()
    mock_showinfo.assert_called_once()
    assert "No common SSH configuration files" in mock_showinfo.call_args[0][1]

def test_cli_ssh_config_flag(monkeypatch):
    # Test CLI integration
    mock_run_cli = MagicMock(return_value=0)
    monkeypatch.setattr("gptscan.run_cli", mock_run_cli)

    mock_get_ssh_paths = MagicMock(return_value=["/fake/ssh_config"])
    monkeypatch.setattr("gptscan.get_ssh_config_paths", mock_get_ssh_paths)

    # Simulate CLI arguments
    test_args = ["gptscan.py", "--ssh-config", "--cli"]
    monkeypatch.setattr("sys.argv", test_args)

    # We need to mock sys.exit to prevent the test from exiting
    mock_exit = MagicMock()
    monkeypatch.setattr("sys.exit", mock_exit)

    # Also mock create_gui just in case
    monkeypatch.setattr("gptscan.create_gui", MagicMock())

    gptscan.main()

    # run_cli should be called with /fake/ssh_config in targets
    assert mock_run_cli.called
    args, kwargs = mock_run_cli.call_args
    targets = args[0]
    assert "/fake/ssh_config" in targets
