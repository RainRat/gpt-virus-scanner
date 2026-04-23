import pytest
from unittest.mock import MagicMock
import gptscan

def test_scan_git_diff_click_success(monkeypatch):
    # Setup mocks
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "/test/path"
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox, raising=False)

    mock_get_git_diff = MagicMock(return_value="fake diff content")
    monkeypatch.setattr(gptscan, 'get_git_diff', mock_get_git_diff)

    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, 'button_click', mock_button_click)

    # Action
    gptscan.scan_git_diff_click()

    # Assert
    mock_get_git_diff.assert_called_once_with("/test/path")
    mock_button_click.assert_called_once_with(extra_snippets=[("[Git Diff]", b"fake diff content")])

def test_scan_git_diff_click_default_path(monkeypatch):
    # Setup mocks with empty textbox
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = ""
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox, raising=False)

    mock_get_git_diff = MagicMock(return_value="fake diff content")
    monkeypatch.setattr(gptscan, 'get_git_diff', mock_get_git_diff)

    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, 'button_click', mock_button_click)

    # Action
    gptscan.scan_git_diff_click()

    # Assert
    mock_get_git_diff.assert_called_once_with(".")
    mock_button_click.assert_called_once_with(extra_snippets=[("[Git Diff]", b"fake diff content")])

def test_scan_git_diff_click_no_textbox(monkeypatch):
    # Setup mocks with textbox as None
    monkeypatch.setattr(gptscan, 'textbox', None, raising=False)

    mock_get_git_diff = MagicMock(return_value="fake diff content")
    monkeypatch.setattr(gptscan, 'get_git_diff', mock_get_git_diff)

    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, 'button_click', mock_button_click)

    # Action
    gptscan.scan_git_diff_click()

    # Assert
    mock_get_git_diff.assert_called_once_with(".")
    mock_button_click.assert_called_once_with(extra_snippets=[("[Git Diff]", b"fake diff content")])

def test_scan_git_diff_click_no_changes(monkeypatch):
    # Setup mocks
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "."
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox, raising=False)

    mock_get_git_diff = MagicMock(return_value="")
    monkeypatch.setattr(gptscan, 'get_git_diff', mock_get_git_diff)

    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_msgbox)

    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, 'button_click', mock_button_click)

    # Action
    gptscan.scan_git_diff_click()

    # Assert
    mock_get_git_diff.assert_called_once()
    mock_msgbox.showinfo.assert_called_once_with("Git Diff", "No Git changes detected (staged or unstaged) in the target path.")
    mock_button_click.assert_not_called()

def test_scan_git_diff_click_error(monkeypatch):
    # Setup mocks
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "."
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox, raising=False)

    mock_get_git_diff = MagicMock(side_effect=Exception("Git error"))
    monkeypatch.setattr(gptscan, 'get_git_diff', mock_get_git_diff)

    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_msgbox)

    # Action
    gptscan.scan_git_diff_click()

    # Assert
    mock_msgbox.showwarning.assert_called_once_with("Git Diff Error", "Could not retrieve Git diff: Git error")
