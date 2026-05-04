import pytest
from unittest.mock import MagicMock, patch
import gptscan

def test_scan_git_revision_click_success(monkeypatch):
    """Test successful revision scan flow."""
    # Mock textbox
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "/repo/path"
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox, raising=False)

    # Mock git info (found repo)
    monkeypatch.setattr(gptscan, '_get_git_info', lambda path: ("/repo", "path"))

    # Mock user input for revision
    monkeypatch.setattr(gptscan.simpledialog, 'askstring', lambda title, prompt: "HEAD~1")

    # Mock changed files
    mock_changed = ["/repo/file1.py", "/repo/file2.py"]
    monkeypatch.setattr(gptscan, 'get_git_changed_files', MagicMock(return_value=mock_changed))

    # Mock _set_scan_target and button_click
    mock_set_target = MagicMock()
    monkeypatch.setattr(gptscan, '_set_scan_target', mock_set_target)
    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, 'button_click', mock_button_click)

    # Action
    gptscan.scan_git_revision_click()

    # Assert
    gptscan.get_git_changed_files.assert_called_once_with("/repo/path", ref="HEAD~1")
    mock_set_target.assert_called_once_with(mock_changed)
    mock_button_click.assert_called_once()

def test_scan_git_revision_click_no_repo(monkeypatch):
    """Test when target path is not in a git repository."""
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "/not/a/repo"
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox, raising=False)

    # Mock git info (not found)
    monkeypatch.setattr(gptscan, '_get_git_info', lambda path: (None, None))

    # Mock showwarning
    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_msgbox)

    # Action
    gptscan.scan_git_revision_click()

    # Assert
    mock_msgbox.showwarning.assert_called_once_with("Git Error", "Target path is not part of a Git repository.")

def test_scan_git_revision_click_cancel(monkeypatch):
    """Test when user cancels the revision dialog."""
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "/repo"
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox, raising=False)
    monkeypatch.setattr(gptscan, '_get_git_info', lambda path: ("/repo", "."))

    # Mock user cancel
    monkeypatch.setattr(gptscan.simpledialog, 'askstring', lambda title, prompt: None)

    mock_get_git = MagicMock()
    monkeypatch.setattr(gptscan, 'get_git_changed_files', mock_get_git)

    # Action
    gptscan.scan_git_revision_click()

    # Assert
    mock_get_git.assert_not_called()

def test_scan_git_revision_click_no_files(monkeypatch):
    """Test when no files are changed in the given revision."""
    monkeypatch.setattr(gptscan, 'textbox', MagicMock(get=lambda: "."), raising=False)
    monkeypatch.setattr(gptscan, '_get_git_info', lambda path: ("/repo", "."))
    monkeypatch.setattr(gptscan.simpledialog, 'askstring', lambda title, prompt: "main")

    # Mock no changed files
    monkeypatch.setattr(gptscan, 'get_git_changed_files', MagicMock(return_value=[]))

    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_msgbox)
    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, 'button_click', mock_button_click)

    # Action
    gptscan.scan_git_revision_click()

    # Assert
    mock_msgbox.showinfo.assert_called_once_with("Git Revision", "No changed files found for revision 'main'.")
    mock_button_click.assert_not_called()

def test_scan_git_revision_click_error(monkeypatch):
    """Test exception handling in the revision scan flow."""
    monkeypatch.setattr(gptscan, 'textbox', MagicMock(get=lambda: "."), raising=False)

    # Trigger exception
    monkeypatch.setattr(gptscan, '_get_git_info', MagicMock(side_effect=Exception("Unexpected error")))

    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_msgbox)

    # Action
    gptscan.scan_git_revision_click()

    # Assert
    mock_msgbox.showwarning.assert_called_once()
    assert "Could not scan Git revision" in mock_msgbox.showwarning.call_args[0][1]
