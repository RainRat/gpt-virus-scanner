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

    # Mock Entry
    class MockEntry:
        def __init__(self, *args, **kwargs):
            self.val = ""
        def delete(self, start, end): self.val = ""
        def insert(self, idx, val): self.val = val
        def get(self): return self.val
        def config(self, **kwargs): pass
        def grid(self, **kwargs): pass
        def pack(self, **kwargs): pass
        def cget(self, key): return ""

    monkeypatch.setattr(gptscan.ttk, 'Entry', MockEntry)

    # Mock tk.Label/ttk.Label
    class MockLabel:
        def __init__(self, *args, **kwargs):
            self.config_data = {}
        def config(self, **kwargs): self.config_data.update(kwargs)
        def cget(self, key): return self.config_data.get(key, "")
        def grid(self, **kwargs): pass
        def pack(self, **kwargs): pass
        def grid_forget(self): pass
        def pack_forget(self): pass
        def winfo_viewable(self): return True

    monkeypatch.setattr(gptscan.tk, 'Label', MockLabel)
    monkeypatch.setattr(gptscan.ttk, 'Label', MockLabel)

    # Mock ScrolledText
    class MockScrolledText:
        def __init__(self, *args, **kwargs):
            self.content = ""
            self.tags = []
        def delete(self, start, end): self.content = ""
        def insert(self, idx, val): self.content += val
        def get(self, start, end): return self.content
        def config(self, **kwargs): pass
        def pack(self, **kwargs): pass
        def pack_forget(self): pass
        def tag_add(self, tag, start, end): self.tags.append((tag, start, end))
        def tag_configure(self, *args, **kwargs): pass
        def see(self, *args): pass
        def winfo_viewable(self): return True

    monkeypatch.setattr(gptscan.scrolledtext, 'ScrolledText', MockScrolledText)

    # Mock messagebox
    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_msgbox)

    # Capture components from gptscan.view_details
    captured = {
        'labels': [],
        'scrolledtexts': []
    }

    original_label = gptscan.tk.Label
    def mock_label_init(*args, **kwargs):
        lbl = MockLabel(*args, **kwargs)
        captured['labels'].append(lbl)
        return lbl
    monkeypatch.setattr(gptscan.tk, 'Label', mock_label_init)
    monkeypatch.setattr(gptscan.ttk, 'Label', mock_label_init)

    original_st = gptscan.scrolledtext.ScrolledText
    def mock_st_init(*args, **kwargs):
        st = MockScrolledText(*args, **kwargs)
        captured['scrolledtexts'].append(st)
        return st
    monkeypatch.setattr(gptscan.scrolledtext, 'ScrolledText', mock_st_init)

    def mock_button_init(master, **kwargs):
        btn = MagicMock()
        text = kwargs.get('text', '')
        if text:
            captured[f"btn_{text}"] = (btn, kwargs.get('command'))
        return btn
    monkeypatch.setattr(gptscan.ttk, 'Button', mock_button_init)

    def mock_labelframe_init(master, **kwargs):
        lf = MagicMock()
        text = kwargs.get('text', '')
        if text:
            captured[f"lf_{text}"] = lf
        return lf
    monkeypatch.setattr(gptscan.ttk, 'LabelFrame', mock_labelframe_init)

    return captured, mock_msgbox, mock_tree, mock_toplevel

def setup_details(captured_env, item_id, path, own_conf="90%", admin="Admin", user="User", gpt_conf="80%", snippet="snippet", line=1):
    captured, mock_msgbox, mock_tree, mock_toplevel = captured_env
    raw_vals = [path, own_conf, admin, user, gpt_conf, snippet, line]
    mock_tree._item_values[item_id] = [path, own_conf, admin, user, gpt_conf, snippet, line, json.dumps(raw_vals)]
    mock_tree.get_children.return_value = list(mock_tree._item_values.keys())
    mock_tree.selection.return_value = [item_id]
    gptscan.view_details(item_id=item_id)
    return captured

def test_view_details_open_window(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "test.py", admin="Admin Notes", user="User Notes", snippet="snippet content")

    gptscan.tk.Toplevel.assert_called_once()
    mock_toplevel.title.assert_called_with("Result 1 of 1 - test.py")

    # Verify risk_badge (it's the only one with HIGH RISK text)
    risk_badges = [l for l in captured['labels'] if l.cget('text') == "HIGH RISK"]
    assert len(risk_badges) == 1
    assert risk_badges[0].cget('background') == "#ffcccc"

    # Verify notes and snippet
    all_content = " ".join([st.content for st in captured['scrolledtexts']])
    assert "Admin Notes" in all_content
    assert "User Notes" in all_content
    assert "snippet content" in all_content

def test_view_details_low_risk(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "safe.py", own_conf="10%", admin="", user="", gpt_conf="", snippet="safe")

    risk_badges = [l for l in captured['labels'] if l.cget('text') == "LOW RISK"]
    assert len(risk_badges) == 1
    assert risk_badges[0].cget('background') == "lightgrey"

def test_view_details_no_selection(mock_view_details_env, monkeypatch):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    mock_tree.selection.return_value = []
    gptscan.tk.Toplevel.reset_mock()
    gptscan.view_details()
    gptscan.tk.Toplevel.assert_not_called()

def test_view_details_navigation(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    raw1 = ["file1.py", "10%", "", "", "", "snippet1", 1]
    mock_tree._item_values["item1"] = ["file1.py", "10%", "", "", "", "snippet1", 1, json.dumps(raw1)]
    raw2 = ["file2.py", "20%", "Admin", "User", "90%", "snippet2", 1]
    mock_tree._item_values["item2"] = ["file2.py", "20%", "Admin", "User", "90%", "snippet2", 1, json.dumps(raw2)]
    raw3 = ["file3.py", "30%", "", "", "", "snippet3", 1]
    mock_tree._item_values["item3"] = ["file3.py", "30%", "", "", "", "snippet3", 1, json.dumps(raw3)]
    mock_tree.get_children.return_value = ["item1", "item2", "item3"]

    gptscan.view_details(item_id="item2")
    assert mock_toplevel.title.call_args_list[-1][0][0] == "Result 2 of 3 - file2.py"

    next_cmd = captured["btn_Next >"][1]
    next_cmd()
    mock_tree.selection_set.assert_called_with("item3")
    assert mock_toplevel.title.call_args_list[-1][0][0] == "Result 3 of 3 - file3.py"

    prev_cmd = captured["btn_< Previous"][1]
    prev_cmd()
    mock_tree.selection_set.assert_called_with("item2")
    assert mock_toplevel.title.call_args_list[-1][0][0] == "Result 2 of 3 - file2.py"

def test_view_details_keyboard_bindings(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    raw1 = ["file1.py", "10%", "", "", "", "snippet1", 1]
    mock_tree._item_values["item1"] = ["file1.py", "10%", "", "", "", "snippet1", 1, json.dumps(raw1)]
    raw2 = ["file2.py", "20%", "Admin", "User", "90%", "snippet2", 1]
    mock_tree._item_values["item2"] = ["file2.py", "20%", "Admin", "User", "90%", "snippet2", 1, json.dumps(raw2)]
    mock_tree.get_children.return_value = ["item1", "item2"]

    captured_bindings = {}
    mock_toplevel.bind.side_effect = lambda event, func: captured_bindings.update({event: func})

    gptscan.view_details(item_id="item1")
    assert '<Left>' in captured_bindings
    assert '<Right>' in captured_bindings

    captured_bindings['<Right>'](None)
    mock_tree.selection_set.assert_called_with("item2")
    assert mock_toplevel.title.call_args_list[-1][0][0] == "Result 2 of 2 - file2.py"

def test_toggle_source_success(mock_view_details_env, monkeypatch):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "test.py", snippet="snippet content", line=5)

    toggle_cmd = captured["btn_Show Full Source"][1]
    btn_mock = captured["btn_Show Full Source"][0]
    monkeypatch.setattr(os.path, "exists", lambda x: True)
    monkeypatch.setattr(os.path, "getsize", lambda x: 100)

    with patch("builtins.open", mock_open(read_data="full content")):
        toggle_cmd()

    btn_mock.config.assert_called_with(text="Show Snippet")
    st = captured['scrolledtexts'][-1]
    assert "full content" in st.content
    assert ("highlight", "5.0", "5.end") in st.tags

    toggle_cmd()
    btn_mock.config.assert_called_with(text="Show Full Source")
    assert "snippet content" in st.content

def test_toggle_source_virtual_file(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "[Clipboard]", snippet="snippet")
    toggle_cmd = captured["btn_Show Full Source"][1]
    toggle_cmd()
    mock_msgbox.showinfo.assert_called_with("Full Source", "Full source is not available for files inside archives, web links, or clipboard content.")

def test_toggle_source_missing_file(mock_view_details_env, monkeypatch):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "missing.py", snippet="snippet")
    toggle_cmd = captured["btn_Show Full Source"][1]
    monkeypatch.setattr(os.path, "exists", lambda x: False)
    toggle_cmd()
    mock_msgbox.showerror.assert_called_with("Error", "File not found: missing.py")

def test_toggle_source_large_file_cancel(mock_view_details_env, monkeypatch):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "large.py", snippet="snippet")
    toggle_cmd = captured["btn_Show Full Source"][1]
    monkeypatch.setattr(os.path, "exists", lambda x: True)
    monkeypatch.setattr(os.path, "getsize", lambda x: 3 * 1024 * 1024)
    mock_msgbox.askyesno.return_value = False
    toggle_cmd()
    mock_msgbox.askyesno.assert_called()

def test_toggle_source_read_error(mock_view_details_env, monkeypatch):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "error.py", snippet="snippet")
    toggle_cmd = captured["btn_Show Full Source"][1]
    monkeypatch.setattr(os.path, "exists", lambda x: True)
    monkeypatch.setattr(os.path, "getsize", lambda x: 100)
    with patch("builtins.open", side_effect=OSError("Read error")):
        toggle_cmd()
    mock_msgbox.showerror.assert_called_with("Error", "Could not read file: Read error")

def test_view_details_exclude_navigation(mock_view_details_env, monkeypatch):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    raw1 = ["file1.py", "10%", "", "", "", "snippet1", 1]
    mock_tree._item_values["item1"] = ["file1.py", "10%", "", "", "", "snippet1", 1, json.dumps(raw1)]
    raw2 = ["file2.py", "20%", "Admin", "User", "90%", "snippet2", 1]
    mock_tree._item_values["item2"] = ["file2.py", "20%", "Admin", "User", "90%", "snippet2", 1, json.dumps(raw2)]
    mock_tree.get_children.return_value = ["item1", "item2"]

    gptscan.view_details(item_id="item1")
    exclude_cmd = captured["btn_Exclude"][1]
    monkeypatch.setattr(gptscan, "exclude_paths", MagicMock(return_value=True))
    mock_tree.get_children.return_value = ["item2"]
    exclude_cmd()

    mock_tree.selection_set.assert_called_with("item2")
    assert mock_toplevel.title.call_args_list[-1][0][0] == "Result 1 of 1 - file2.py"

    mock_tree.get_children.return_value = []
    exclude_cmd()
    mock_toplevel.destroy.assert_called_once()

def test_view_details_analyze_now(mock_view_details_env, monkeypatch):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "test.py", admin="", user="", gpt_conf="")

    analyze_cmd = captured["btn_Analyze with AI"][1]
    mock_result = {"administrator": "New Admin Info", "end-user": "New User Info", "threat-level": 95}

    monkeypatch.setattr(gptscan, "request_single_gpt_analysis", MagicMock(return_value=mock_result))
    monkeypatch.setattr(gptscan.Config, "GPT_ENABLED", True)
    monkeypatch.setattr(gptscan.threading.Thread, "start", lambda self: self._target(*self._args, **self._kwargs))
    monkeypatch.setattr(gptscan, "enqueue_ui_update", lambda func, *args, **kwargs: func(*args, **kwargs))

    mock_update_tree = MagicMock(side_effect=lambda item_id, values: mock_tree._item_values.update({item_id: values}))
    monkeypatch.setattr(gptscan, "update_tree_row", mock_update_tree)

    analyze_cmd()
    gptscan.request_single_gpt_analysis.assert_called_once()

    item1_vals = mock_tree._item_values["item1"]
    assert item1_vals[2] == "New Admin Info"
    assert item1_vals[3] == "New User Info"

def test_view_details_persistent_full_source(mock_view_details_env, monkeypatch):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    raw1 = ["file1.py", "10%", "", "", "", "snippet1", 1]
    mock_tree._item_values["item1"] = ["file1.py", "10%", "", "", "", "snippet1", 1, json.dumps(raw1)]
    raw2 = ["file2.py", "20%", "", "", "", "snippet2", 1]
    mock_tree._item_values["item2"] = ["file2.py", "20%", "", "", "", "snippet2", 1, json.dumps(raw2)]

    mock_tree.get_children.return_value = ["item1", "item2"]
    monkeypatch.setattr(os.path, "exists", lambda x: True)
    monkeypatch.setattr(os.path, "getsize", lambda x: 100)

    gptscan.view_details(item_id="item1")
    toggle_cmd = captured["btn_Show Full Source"][1]

    with patch("builtins.open", mock_open(read_data="full content 1")):
        toggle_cmd()

    st = captured['scrolledtexts'][-1]
    assert "full content 1" in st.content

    next_cmd = captured["btn_Next >"][1]
    with patch("builtins.open", mock_open(read_data="full content 2")):
        next_cmd()
    assert "full content 2" in st.content

def test_view_details_copy_path(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "test.py")
    copy_path_cmd = captured["btn_Copy"][1]
    from gptscan import root as mock_root
    copy_path_cmd()
    mock_root.clipboard_clear.assert_called_once()
    mock_root.clipboard_append.assert_called_with("test.py")
    # Verified feedback via status bar
    # status_bar is the first label created in view_details
    status_bar = captured['labels'][0]
    assert status_bar.config_data.get('text') == "File path copied to clipboard."

def test_view_details_copy_analysis(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "test.py", own_conf="90%", admin="Admin Notes", user="User Notes", gpt_conf="80%", snippet="print('test')")
    from gptscan import root as mock_root
    copy_cmd = captured["btn_Copy Analysis"][1]
    copy_cmd()
    mock_root.clipboard_clear.assert_called_once()
    copied_text = mock_root.clipboard_append.call_args[0][0]
    assert "Path: test.py" in copied_text
    assert "Local Threat Level: 90%" in copied_text
    assert "AI Threat Level: 80%" in copied_text
    assert "Admin Notes:\nAdmin Notes" in copied_text
    assert "Snippet:\nprint('test')" in copied_text
    # Verified feedback via status bar
    # status_bar is the first label created in view_details
    status_bar = captured['labels'][0]
    assert status_bar.config_data.get('text') == "Detailed analysis copied to clipboard."

def test_view_details_refresh_content_missing_id(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    raw1 = ["file1.py", "10%", "", "", "", "snippet1", 1]
    mock_tree._item_values["item1"] = ["file1.py", "10%", "", "", "", "snippet1", 1, json.dumps(raw1)]
    mock_tree.get_children.return_value = []
    gptscan.view_details(item_id="item1")
    mock_toplevel.title.assert_called_with("Result Details - file1.py")

def test_view_details_rescan(mock_view_details_env, monkeypatch):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "test.py", own_conf="50%", snippet="old snippet")

    rescan_cmd = captured["btn_Rescan"][1]
    rescan_btn_mock = captured["btn_Rescan"][0]

    # Mock run_rescan to simulate a successful rescan
    def mock_run_rescan(paths, item_map, settings, cancel_event):
        target_id = item_map[paths[0]]
        new_vals = ["test.py", "100%", "New Admin", "New User", "99%", "new snippet", 1]
        gptscan.update_tree_row(target_id, tuple(new_vals))
        gptscan.finish_scan_state(1, 1)

    monkeypatch.setattr(gptscan, "run_rescan", mock_run_rescan)
    monkeypatch.setattr(gptscan.threading.Thread, "start", lambda self: self._target(*self._args, **self._kwargs))
    monkeypatch.setattr(gptscan, "enqueue_ui_update", lambda func, *args, **kwargs: func(*args, **kwargs))

    mock_update_tree = MagicMock(side_effect=lambda item_id, values: mock_tree._item_values.update({item_id: values}))
    monkeypatch.setattr(gptscan, "update_tree_row", mock_update_tree)

    rescan_cmd()

    mock_update_tree.assert_called()
    assert gptscan.current_cancel_event is None

    # Verify content was updated in scrolledtexts
    all_content = " ".join([st.content for st in captured['scrolledtexts']])
    assert "new snippet" in all_content
    assert "New Admin" in all_content
    assert "New User" in all_content

    # Verify button states were toggled
    rescan_btn_mock.config.assert_any_call(state='disabled', text='Rescanning...')
    rescan_btn_mock.config.assert_any_call(state='normal', text='Rescan')
