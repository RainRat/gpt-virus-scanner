import os
import pytest
from unittest.mock import MagicMock, patch
import gptscan

def test_browse_file_list_click_success(monkeypatch, tmp_path):
    # Mock filedialog.askopenfilename to return a temporary file path
    list_file = tmp_path / "list.txt"
    list_file.write_text("file1.py\nfile2.js\n\n  folder/  \n", encoding='utf-8')

    monkeypatch.setattr("gptscan.filedialog.askopenfilename", lambda **kwargs: str(list_file))

    # Mock _set_scan_target to capture the paths
    mock_set_target = MagicMock()
    monkeypatch.setattr("gptscan._set_scan_target", mock_set_target)

    gptscan.browse_file_list_click()

    # Verify that _set_scan_target was called with the expected paths
    mock_set_target.assert_called_once_with(["file1.py", "file2.js", "folder/"])

def test_browse_file_list_click_cancel(monkeypatch):
    # Mock filedialog.askopenfilename to return empty string (cancel)
    monkeypatch.setattr("gptscan.filedialog.askopenfilename", lambda **kwargs: "")

    mock_set_target = MagicMock()
    monkeypatch.setattr("gptscan._set_scan_target", mock_set_target)

    gptscan.browse_file_list_click()

    # Verify that _set_scan_target was not called
    mock_set_target.assert_not_called()

def test_browse_file_list_click_error(monkeypatch, tmp_path):
    # Mock filedialog.askopenfilename to return a path that doesn't exist or causes error
    monkeypatch.setattr("gptscan.filedialog.askopenfilename", lambda **kwargs: "non_existent.txt")

    mock_messagebox = MagicMock()
    monkeypatch.setattr("gptscan.messagebox.showerror", mock_messagebox)

    gptscan.browse_file_list_click()

    # Verify that showerror was called
    mock_messagebox.assert_called_once()
    assert "Could not read file list" in mock_messagebox.call_args[0][1]
