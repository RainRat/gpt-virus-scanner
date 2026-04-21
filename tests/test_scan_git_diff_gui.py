from unittest.mock import MagicMock
import pytest
import gptscan

def test_scan_git_diff_click_success(monkeypatch):
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "/path/to/repo"
    monkeypatch.setattr(gptscan, "textbox", mock_textbox, raising=False)

    mock_get_git_diff = MagicMock(return_value="fake diff content")
    monkeypatch.setattr(gptscan, "get_git_diff", mock_get_git_diff)

    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, "button_click", mock_button_click)

    gptscan.scan_git_diff_click()

    mock_get_git_diff.assert_called_once_with("/path/to/repo")
    mock_button_click.assert_called_once_with(extra_snippets=[("[Git Diff]", b"fake diff content")])

def test_scan_git_diff_click_no_changes(monkeypatch):
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "."
    monkeypatch.setattr(gptscan, "textbox", mock_textbox, raising=False)

    mock_get_git_diff = MagicMock(return_value="")
    monkeypatch.setattr(gptscan, "get_git_diff", mock_get_git_diff)

    mock_messagebox = MagicMock()
    monkeypatch.setattr(gptscan, "messagebox", mock_messagebox)

    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, "button_click", mock_button_click)

    gptscan.scan_git_diff_click()

    mock_get_git_diff.assert_called_once_with(".")
    mock_messagebox.showinfo.assert_called_once_with("Git Diff", "No Git changes detected (staged or unstaged) in the target path.")
    mock_button_click.assert_not_called()

def test_scan_git_diff_click_exception(monkeypatch):
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "."
    monkeypatch.setattr(gptscan, "textbox", mock_textbox, raising=False)

    mock_get_git_diff = MagicMock(side_effect=Exception("Git error"))
    monkeypatch.setattr(gptscan, "get_git_diff", mock_get_git_diff)

    mock_messagebox = MagicMock()
    monkeypatch.setattr(gptscan, "messagebox", mock_messagebox)

    gptscan.scan_git_diff_click()

    mock_messagebox.showwarning.assert_called_once_with("Git Diff Error", "Could not retrieve Git diff: Git error")

def test_scan_git_diff_click_default_path(monkeypatch):
    # Test when textbox is None
    monkeypatch.setattr(gptscan, "textbox", None, raising=False)

    mock_get_git_diff = MagicMock(return_value="some diff")
    monkeypatch.setattr(gptscan, "get_git_diff", mock_get_git_diff)

    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, "button_click", mock_button_click)

    gptscan.scan_git_diff_click()

    mock_get_git_diff.assert_called_once_with(".")
    mock_button_click.assert_called_once()

def test_scan_git_diff_click_empty_textbox(monkeypatch):
    # Test when textbox.get() returns empty string
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "  "
    monkeypatch.setattr(gptscan, "textbox", mock_textbox, raising=False)

    mock_get_git_diff = MagicMock(return_value="some diff")
    monkeypatch.setattr(gptscan, "get_git_diff", mock_get_git_diff)

    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, "button_click", mock_button_click)

    gptscan.scan_git_diff_click()

    mock_get_git_diff.assert_called_once_with(".")
    mock_button_click.assert_called_once()
