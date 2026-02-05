import os
import threading
from unittest.mock import MagicMock, patch
import pytest
import gptscan

def test_button_click_with_git_changes_enabled(monkeypatch):
    # Setup mocks
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "/some/repo"
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox, raising=False)

    monkeypatch.setattr(gptscan, 'current_cancel_event', None)

    mock_git_var = MagicMock()
    mock_git_var.get.return_value = True
    monkeypatch.setattr(gptscan, 'git_var', mock_git_var, raising=False)

    # Mock other vars
    monkeypatch.setattr(gptscan, 'deep_var', MagicMock(get=lambda: False), raising=False)
    monkeypatch.setattr(gptscan, 'all_var', MagicMock(get=lambda: False), raising=False)
    monkeypatch.setattr(gptscan, 'gpt_var', MagicMock(get=lambda: False), raising=False)
    monkeypatch.setattr(gptscan, 'dry_var', MagicMock(get=lambda: True), raising=False) # Dry run to avoid model load

    # Mock tree
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)

    # Mock status_label and progress_bar
    monkeypatch.setattr(gptscan, 'status_label', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'progress_bar', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'root', MagicMock(), raising=False)

    # Mock buttons
    monkeypatch.setattr(gptscan, 'scan_button', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'cancel_button', MagicMock(), raising=False)

    # Mock get_git_changed_files
    mock_get_git = MagicMock(return_value=["/some/repo/file1.py", "/some/repo/file2.py"])
    monkeypatch.setattr(gptscan, 'get_git_changed_files', mock_get_git)

    # Mock Thread
    mock_thread = MagicMock()
    monkeypatch.setattr(gptscan.threading, 'Thread', mock_thread)

    # Action
    gptscan.button_click()

    # Assert
    mock_get_git.assert_called_once_with("/some/repo")
    mock_thread.assert_called_once()
    _, kwargs = mock_thread.call_args
    assert kwargs['args'][0] == ["/some/repo/file1.py", "/some/repo/file2.py"]

def test_button_click_with_git_changes_no_changes(monkeypatch):
    # Setup mocks
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "/some/repo"
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox, raising=False)

    monkeypatch.setattr(gptscan, 'current_cancel_event', None)

    mock_git_var = MagicMock()
    mock_git_var.get.return_value = True
    monkeypatch.setattr(gptscan, 'git_var', mock_git_var, raising=False)

    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_msgbox)

    # Mock get_git_changed_files to return empty list
    mock_get_git = MagicMock(return_value=[])
    monkeypatch.setattr(gptscan, 'get_git_changed_files', mock_get_git)

    # Mock Thread (should not be called)
    mock_thread = MagicMock()
    monkeypatch.setattr(gptscan.threading, 'Thread', mock_thread)

    # Action
    gptscan.button_click()

    # Assert
    mock_get_git.assert_called_once_with("/some/repo")
    mock_msgbox.showinfo.assert_called_once_with("Git Scan", "No git changes detected in the selected directory.")
    mock_thread.assert_not_called()
