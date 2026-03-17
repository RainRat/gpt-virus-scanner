import pytest
from unittest.mock import MagicMock, patch, mock_open, ANY
import gptscan
import json
import tkinter as tk
import os

@pytest.fixture
def mock_view_details_env(monkeypatch):
    """Setup a mock environment for view_details tests."""
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
    mock_tree.exists.side_effect = lambda iid: iid in mock_tree._item_values

    mock_root = MagicMock()
    monkeypatch.setattr(gptscan, 'root', mock_root)

    # Mock Toplevel
    mock_toplevel = MagicMock()
    monkeypatch.setattr(gptscan.tk, 'Toplevel', MagicMock(return_value=mock_toplevel))

    # Mock tk.Label for risk_badge
    mock_label = MagicMock()
    monkeypatch.setattr(gptscan.tk, 'Label', MagicMock(return_value=mock_label))

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

    return captured, mock_st, mock_msgbox, mock_tree, mock_toplevel, mock_label

def setup_details(captured_env, item_id, path, own_conf="90%", admin="Admin", user="User", gpt_conf="80%", snippet="snippet", line=1):
    captured, mock_st, mock_msgbox, mock_tree, mock_toplevel, mock_label = captured_env
    raw_vals = [path, own_conf, admin, user, gpt_conf, snippet, line]
    mock_tree._item_values[item_id] = [path, own_conf, admin, user, gpt_conf, snippet, line, json.dumps(raw_vals)]
    mock_tree.get_children.return_value = list(mock_tree._item_values.keys())
    mock_tree.selection.return_value = [item_id]
    gptscan.view_details(item_id=item_id)
    return captured

def test_view_details_open_window(mock_view_details_env):
    captured, mock_st, mock_msgbox, mock_tree, mock_toplevel, mock_label = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "test.py")

    # Verify Toplevel was created
    gptscan.tk.Toplevel.assert_called_once()
    mock_toplevel.title.assert_called_with("Result 1 of 1 - test.py")

    # Verify risk_badge was created and configured correctly (90% own_conf -> HIGH RISK)
    mock_label.config.assert_any_call(text="HIGH RISK", background="#ffcccc", foreground="darkred")

    # Verify data was inserted into ScrolledText widgets
    # Check that the data was inserted
    calls = mock_st.insert.call_args_list
    contents = [call[0][1] for call in calls]
    assert "Admin" in contents
    assert "User" in contents
    assert "snippet" in contents

def test_view_details_low_risk(mock_view_details_env):
    setup_details(mock_view_details_env, "item1", "safe.py", own_conf="10%", admin="", user="", gpt_conf="", snippet="safe")
    _, _, _, _, _, mock_label = mock_view_details_env
    mock_label.config.assert_any_call(text="LOW RISK", background="lightgrey", foreground="grey")

def test_view_details_no_selection(mock_view_details_env, monkeypatch):
    captured, mock_st, mock_msgbox, mock_tree, mock_toplevel, mock_label = mock_view_details_env
    mock_tree.selection.return_value = []

    # We need to reset the mock because it might have been called by setup_details if we used it
    gptscan.tk.Toplevel.reset_mock()

    gptscan.view_details()

    gptscan.tk.Toplevel.assert_not_called()

def test_view_details_navigation(mock_view_details_env):
    captured, mock_st, mock_msgbox, mock_tree, mock_toplevel, mock_label = mock_view_details_env

    # Setup 3 items
    raw1 = ["file1.py", "10%", "", "", "", "snippet1", 1]
    mock_tree._item_values["item1"] = ["file1.py", "10%", "", "", "", "snippet1", 1, json.dumps(raw1)]
    raw2 = ["file2.py", "20%", "Admin", "User", "90%", "snippet2", 1]
    mock_tree._item_values["item2"] = ["file2.py", "20%", "Admin", "User", "90%", "snippet2", 1, json.dumps(raw2)]
    raw3 = ["file3.py", "30%", "", "", "", "snippet3", 1]
    mock_tree._item_values["item3"] = ["file3.py", "30%", "", "", "", "snippet3", 1, json.dumps(raw3)]

    mock_tree.get_children.return_value = ["item1", "item2", "item3"]

    gptscan.view_details(item_id="item2")

    # Verify initial state
    assert mock_toplevel.title.call_args_list[-1][0][0] == "Result 2 of 3 - file2.py"

    # Test "Next >" button
    next_cmd = captured["btn_Next >"][1]
    next_cmd()
    mock_tree.selection_set.assert_called_with("item3")
    mock_tree.see.assert_called_with("item3")
    assert mock_toplevel.title.call_args_list[-1][0][0] == "Result 3 of 3 - file3.py"

    # Test "< Previous" button
    prev_cmd = captured["btn_< Previous"][1]
    prev_cmd()
    mock_tree.selection_set.assert_called_with("item2")
    mock_tree.see.assert_called_with("item2")
    assert mock_toplevel.title.call_args_list[-1][0][0] == "Result 2 of 3 - file2.py"

def test_view_details_keyboard_bindings(mock_view_details_env):
    captured, mock_st, mock_msgbox, mock_tree, mock_toplevel, mock_label = mock_view_details_env

    raw1 = ["file1.py", "10%", "", "", "", "snippet1", 1]
    mock_tree._item_values["item1"] = ["file1.py", "10%", "", "", "", "snippet1", 1, json.dumps(raw1)]
    raw2 = ["file2.py", "20%", "Admin", "User", "90%", "snippet2", 1]
    mock_tree._item_values["item2"] = ["file2.py", "20%", "Admin", "User", "90%", "snippet2", 1, json.dumps(raw2)]

    mock_tree.get_children.return_value = ["item1", "item2"]

    # Capture bindings
    captured_bindings = {}
    mock_toplevel.bind.side_effect = lambda event, func: captured_bindings.update({event: func})

    gptscan.view_details(item_id="item1")

    assert '<Left>' in captured_bindings
    assert '<Right>' in captured_bindings

    # Trigger Right arrow
    captured_bindings['<Right>'](None)
    mock_tree.selection_set.assert_called_with("item2")
    mock_tree.see.assert_called_with("item2")
    assert mock_toplevel.title.call_args_list[-1][0][0] == "Result 2 of 2 - file2.py"

def test_toggle_source_success(mock_view_details_env, monkeypatch):
    captured, mock_st, mock_msgbox, mock_tree, mock_toplevel, mock_label = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "test.py", snippet="snippet")

    toggle_cmd = captured["btn_Show Full Source"][1]
    btn_mock = captured["btn_Show Full Source"][0]

    monkeypatch.setattr(os.path, "exists", lambda x: True)
    monkeypatch.setattr(os.path, "getsize", lambda x: 100)

    # Clear mocks after initial view_details population
    mock_st.insert.reset_mock()

    with patch("builtins.open", mock_open(read_data="full content")):
        toggle_cmd()

    mock_st.insert.assert_called_with(ANY, "full content")
    btn_mock.config.assert_called_with(text="Show Snippet")

    mock_st.insert.reset_mock()
    toggle_cmd()
    mock_st.insert.assert_called_with(ANY, "snippet")
    btn_mock.config.assert_called_with(text="Show Full Source")

def test_toggle_source_virtual_file(mock_view_details_env):
    captured, mock_st, mock_msgbox, mock_tree, mock_toplevel, mock_label = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "[Clipboard]", snippet="snippet")

    toggle_cmd = captured["btn_Show Full Source"][1]

    # Clear mocks after initial view_details population
    mock_st.insert.reset_mock()

    toggle_cmd()

    mock_msgbox.showinfo.assert_called_with("Full Source", "Full source is not available for virtual files or clipboard content.")
    mock_st.insert.assert_not_called()

def test_toggle_source_missing_file(mock_view_details_env, monkeypatch):
    captured, mock_st, mock_msgbox, mock_tree, mock_toplevel, mock_label = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "missing.py", snippet="snippet")

    toggle_cmd = captured["btn_Show Full Source"][1]
    monkeypatch.setattr(os.path, "exists", lambda x: False)

    toggle_cmd()

    mock_msgbox.showerror.assert_called_with("Error", "File not found: missing.py")

def test_toggle_source_large_file_cancel(mock_view_details_env, monkeypatch):
    captured, mock_st, mock_msgbox, mock_tree, mock_toplevel, mock_label = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "large.py", snippet="snippet")

    toggle_cmd = captured["btn_Show Full Source"][1]
    monkeypatch.setattr(os.path, "exists", lambda x: True)
    monkeypatch.setattr(os.path, "getsize", lambda x: 3 * 1024 * 1024)
    mock_msgbox.askyesno.return_value = False

    # Clear mocks after initial view_details population
    mock_st.insert.reset_mock()

    toggle_cmd()

    mock_msgbox.askyesno.assert_called()
    mock_st.insert.assert_not_called()

def test_toggle_source_read_error(mock_view_details_env, monkeypatch):
    captured, mock_st, mock_msgbox, mock_tree, mock_toplevel, mock_label = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "error.py", snippet="snippet")

    toggle_cmd = captured["btn_Show Full Source"][1]
    monkeypatch.setattr(os.path, "exists", lambda x: True)
    monkeypatch.setattr(os.path, "getsize", lambda x: 100)

    with patch("builtins.open", side_effect=OSError("Read error")):
        toggle_cmd()

    mock_msgbox.showerror.assert_called_with("Error", "Could not read file: Read error")


def test_view_details_exclude_navigation(mock_view_details_env, monkeypatch):
    captured, mock_st, mock_msgbox, mock_tree, mock_toplevel, mock_label = mock_view_details_env

    # Setup 2 items
    raw1 = ["file1.py", "10%", "", "", "", "snippet1", 1]
    mock_tree._item_values["item1"] = ["file1.py", "10%", "", "", "", "snippet1", 1, json.dumps(raw1)]
    raw2 = ["file2.py", "20%", "Admin", "User", "90%", "snippet2", 1]
    mock_tree._item_values["item2"] = ["file2.py", "20%", "Admin", "User", "90%", "snippet2", 1, json.dumps(raw2)]

    mock_tree.get_children.return_value = ["item1", "item2"]

    gptscan.view_details(item_id="item1")

    exclude_cmd = captured["btn_Exclude"][1]

    # Mock exclude_paths to return True (success)
    monkeypatch.setattr(gptscan, "exclude_paths", MagicMock(return_value=True))

    # After exclusion, item1 is gone from visible children
    mock_tree.get_children.return_value = ["item2"]

    exclude_cmd()

    # Verify it switched to item2
    mock_tree.selection_set.assert_called_with("item2")
    mock_tree.see.assert_called_with("item2")
    assert mock_toplevel.title.call_args_list[-1][0][0] == "Result 1 of 1 - file2.py"

    # Exclude the last item
    mock_tree.get_children.return_value = []
    exclude_cmd()

    # Verify window was destroyed
    mock_toplevel.destroy.assert_called_once()


def test_view_details_analyze_now(mock_view_details_env, monkeypatch):
    captured, mock_st, mock_msgbox, mock_tree, mock_toplevel, mock_label = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "test.py", admin="", user="", gpt_conf="")

    analyze_cmd = captured["btn_Analyze with AI"][1]
    analyze_btn_mock = captured["btn_Analyze with AI"][0]

    mock_result = {
        "administrator": "New Admin Info",
        "end-user": "New User Info",
        "threat-level": 95
    }

    monkeypatch.setattr(gptscan, "request_single_gpt_analysis", MagicMock(return_value=mock_result))
    monkeypatch.setattr(gptscan.Config, "GPT_ENABLED", True)

    # We need to mock threading.Thread to run the target synchronously for testing
    def mock_thread_start(self):
        self._target(*self._args, **self._kwargs)

    monkeypatch.setattr(gptscan.threading.Thread, "start", mock_thread_start)

    # We need to process the UI queue
    def mock_enqueue(func, *args, **kwargs):
        func(*args, **kwargs)

    monkeypatch.setattr(gptscan, "enqueue_ui_update", mock_enqueue)

    mock_update_tree = MagicMock()
    def side_effect(item_id, values):
        # Update our mock state so refresh_content sees the new data
        mock_tree._item_values[item_id] = values
    mock_update_tree.side_effect = side_effect
    monkeypatch.setattr(gptscan, "update_tree_row", mock_update_tree)

    # Reset mock_st.insert to verify updates
    mock_st.insert.reset_mock()

    analyze_cmd()

    # Verify AI was called
    gptscan.request_single_gpt_analysis.assert_called_once()

    # Verify update_tree_row was called with new data
    mock_update_tree.assert_called_once()
    updated_vals = mock_update_tree.call_args[0][1]
    # Check updated_vals in mock_tree._item_values instead because update_tree_row might be called with different format
    item1_vals = mock_tree._item_values["item1"]
    assert item1_vals[2] == "New Admin Info"
    assert item1_vals[3] == "New User Info"
    assert updated_vals[4] == "95%"

    # Verify refresh_content updated the view
    calls = mock_st.insert.call_args_list
    contents = [call[0][1] for call in calls]
    assert "New Admin Info" in contents
    assert "New User Info" in contents
