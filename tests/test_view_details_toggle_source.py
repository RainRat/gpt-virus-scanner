import pytest
from unittest.mock import MagicMock, patch, mock_open, ANY
import gptscan
import json
import tkinter as tk
import os

@pytest.fixture
def mock_view_details_env(monkeypatch):
    """Setup a mock environment for view_details and toggle_source."""
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    # Setup mock_tree.item to handle both single and double argument calls
    mock_tree._item_values = {}
    def mock_item_func(item_id, option=None):
        vals = mock_tree._item_values.get(item_id, [])
        if option == "values":
            return vals
        return {"values": vals}
    mock_tree.item.side_effect = mock_item_func

    mock_root = MagicMock()
    monkeypatch.setattr(gptscan, 'root', mock_root)

    # Mock Toplevel
    mock_toplevel = MagicMock()
    monkeypatch.setattr(gptscan.tk, 'Toplevel', MagicMock(return_value=mock_toplevel))

    # Mock ScrolledText
    mock_st = MagicMock()
    monkeypatch.setattr(gptscan.scrolledtext, 'ScrolledText', MagicMock(return_value=mock_st))

    # Mock messagebox
    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_msgbox)

    # Capture components from gptscan.view_details
    captured = {}

    original_button = gptscan.ttk.Button
    def mock_button_init(master, **kwargs):
        btn = MagicMock()
        text = kwargs.get('text', '')
        if text:
            captured[f"btn_{text}"] = (btn, kwargs.get('command'))
        return btn
    monkeypatch.setattr(gptscan.ttk, 'Button', mock_button_init)

    original_labelframe = gptscan.ttk.LabelFrame
    def mock_labelframe_init(master, **kwargs):
        lf = MagicMock()
        text = kwargs.get('text', '')
        if text:
            captured[f"lf_{text}"] = lf
        return lf
    monkeypatch.setattr(gptscan.ttk, 'LabelFrame', mock_labelframe_init)

    return captured, mock_st, mock_msgbox, mock_tree

def setup_details(captured, mock_tree, item_id, path, snippet, line):
    raw_vals = [path, "90%", "Admin", "User", "80%", snippet, line]
    mock_tree._item_values[item_id] = [path, "90%", "Admin", "User", "80%", snippet, line, json.dumps(raw_vals)]
    mock_tree.exists.return_value = True
    mock_tree.get_children.return_value = [item_id]
    gptscan.view_details(item_id=item_id)
    return captured["btn_Show Full Source"][1]

def test_toggle_source_success(mock_view_details_env, monkeypatch):
    captured, mock_st, mock_msgbox, mock_tree = mock_view_details_env
    toggle_cmd = setup_details(captured, mock_tree, "item1", "test.py", "snippet", 2)

    btn, _ = captured["btn_Show Full Source"]
    snippet_lf = captured["lf_Code Snippet"]

    monkeypatch.setattr(os.path, "exists", lambda x: True)
    monkeypatch.setattr(os.path, "getsize", lambda x: 100)

    # Clear mocks after initial view_details population
    mock_st.insert.reset_mock()

    with patch("builtins.open", mock_open(read_data="full content")):
        toggle_cmd()

    mock_st.insert.assert_called_with(ANY, "full content")
    btn.config.assert_called_with(text="Show Snippet")

    mock_st.insert.reset_mock()
    toggle_cmd()
    mock_st.insert.assert_called_with(ANY, "snippet")
    btn.config.assert_called_with(text="Show Full Source")

def test_toggle_source_virtual_file(mock_view_details_env, monkeypatch):
    captured, mock_st, mock_msgbox, mock_tree = mock_view_details_env
    toggle_cmd = setup_details(captured, mock_tree, "item1", "[Clipboard]", "snippet", 1)

    # Clear mocks after initial view_details population
    mock_st.insert.reset_mock()

    toggle_cmd()

    mock_msgbox.showinfo.assert_called_with("Full Source", "Full source is not available for virtual files or clipboard content.")
    mock_st.insert.assert_not_called()

def test_toggle_source_missing_file(mock_view_details_env, monkeypatch):
    captured, mock_st, mock_msgbox, mock_tree = mock_view_details_env
    toggle_cmd = setup_details(captured, mock_tree, "item1", "missing.py", "snippet", 1)

    monkeypatch.setattr(os.path, "exists", lambda x: False)

    toggle_cmd()

    mock_msgbox.showerror.assert_called_with("Error", "File not found: missing.py")

def test_toggle_source_large_file_cancel(mock_view_details_env, monkeypatch):
    captured, mock_st, mock_msgbox, mock_tree = mock_view_details_env
    toggle_cmd = setup_details(captured, mock_tree, "item1", "large.py", "snippet", 1)

    monkeypatch.setattr(os.path, "exists", lambda x: True)
    monkeypatch.setattr(os.path, "getsize", lambda x: 3 * 1024 * 1024)
    mock_msgbox.askyesno.return_value = False

    # Clear mocks after initial view_details population
    mock_st.insert.reset_mock()

    toggle_cmd()

    mock_msgbox.askyesno.assert_called()
    mock_st.insert.assert_not_called()

def test_toggle_source_read_error(mock_view_details_env, monkeypatch):
    captured, mock_st, mock_msgbox, mock_tree = mock_view_details_env
    toggle_cmd = setup_details(captured, mock_tree, "item1", "error.py", "snippet", 1)

    monkeypatch.setattr(os.path, "exists", lambda x: True)
    monkeypatch.setattr(os.path, "getsize", lambda x: 100)

    with patch("builtins.open", side_effect=OSError("Read error")):
        toggle_cmd()

    mock_msgbox.showerror.assert_called_with("Error", "Could not read file: Read error")
