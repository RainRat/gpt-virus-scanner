import pytest
from unittest.mock import MagicMock, patch
import json
import gptscan
import tkinter as tk
from tkinter import ttk
import tkinter.scrolledtext as scrolledtext

class FakeEntry: pass
class FakeText: pass
class FakeScrolledText: pass

@pytest.fixture
def mock_gui(monkeypatch):
    monkeypatch.setattr(gptscan.ttk, 'Entry', FakeEntry)
    monkeypatch.setattr(gptscan.tk, 'Entry', FakeEntry)
    monkeypatch.setattr(gptscan.tk, 'Text', FakeText)
    monkeypatch.setattr(gptscan.scrolledtext, 'ScrolledText', FakeScrolledText)

    mock_root = MagicMock()
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'root', mock_root)
    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    return mock_root, mock_tree

def test_import_from_clipboard_success(mock_gui, monkeypatch):
    mock_root, mock_tree = mock_gui

    data = [
        {
            "path": "test.py",
            "own_conf": "85%",
            "admin_desc": "Suspicious",
            "end-user_desc": "Don't run",
            "gpt_conf": "90%",
            "snippet": "print('hello')",
            "line": "1"
        }
    ]
    json_data = json.dumps(data)

    mock_root.clipboard_get.return_value = json_data
    # Use something that is NOT a FakeEntry/FakeText
    mock_focused = MagicMock()
    mock_root.focus_get.return_value = mock_focused

    mock_finalize = MagicMock()
    monkeypatch.setattr(gptscan, '_finalize_import', mock_finalize)

    result = gptscan.import_from_clipboard()

    assert result == "break"
    mock_finalize.assert_called_once()
    args, _ = mock_finalize.call_args
    assert args[0][0]["path"] == "test.py"
    assert args[1] == "clipboard"

def test_import_from_clipboard_skipped_on_focus(mock_gui, monkeypatch):
    mock_root, mock_tree = mock_gui

    # Focus is on an Entry
    mock_focused = FakeEntry()
    mock_root.focus_get.return_value = mock_focused

    result = gptscan.import_from_clipboard()

    assert result is None
    mock_root.clipboard_get.assert_not_called()

def test_import_from_clipboard_empty(mock_gui, monkeypatch):
    mock_root, mock_tree = mock_gui
    mock_root.clipboard_get.return_value = ""
    mock_root.focus_get.return_value = MagicMock()

    result = gptscan.import_from_clipboard()
    assert result == "break"

def test_import_from_clipboard_read_error(mock_gui, monkeypatch):
    mock_root, mock_tree = mock_gui
    mock_root.clipboard_get.side_effect = Exception("Read Error")
    mock_root.focus_get.return_value = MagicMock()

    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_msgbox)

    result = gptscan.import_from_clipboard()
    assert result == "break"
    mock_msgbox.showwarning.assert_called_once()
    assert "Read Error" in mock_msgbox.showwarning.call_args[0][1]

def test_import_from_clipboard_parse_error(mock_gui, monkeypatch):
    mock_root, mock_tree = mock_gui
    mock_root.clipboard_get.return_value = "invalid content"
    mock_root.focus_get.return_value = MagicMock()

    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_msgbox)

    # parse_report_content will raise ValueError for "invalid content"
    result = gptscan.import_from_clipboard()
    assert result == "break"
    mock_msgbox.showerror.assert_called_once()
    assert "Import Failed" in mock_msgbox.showerror.call_args[0][0]

def test_import_from_clipboard_empty_results(mock_gui, monkeypatch):
    mock_root, mock_tree = mock_gui
    mock_root.clipboard_get.return_value = "[]"
    mock_root.focus_get.return_value = MagicMock()

    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_msgbox)

    result = gptscan.import_from_clipboard()
    assert result == "break"
    mock_msgbox.showwarning.assert_called_once()
    assert "No valid scan results found" in mock_msgbox.showwarning.call_args[0][1]

def test_import_from_clipboard_no_tree(monkeypatch):
    monkeypatch.setattr(gptscan, 'tree', None)
    result = gptscan.import_from_clipboard()
    assert result == "break"
