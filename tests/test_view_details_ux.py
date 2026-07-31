import pytest
import json
import tkinter as tk
from unittest.mock import MagicMock, patch
import gptscan
from tests.test_view_details import mock_view_details_env, setup_details

def test_view_details_copy_code(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    # Setup a result
    raw = ["test.py", "90%", "Admin", "User", "80%", "print('hello')", 1]
    mock_tree._item_values["item1"] = ["test.py", "90%", "Admin", "User", "80%", "print('hello')", 1, json.dumps(raw)]
    mock_tree.get_children.return_value = ["item1"]
    mock_tree.selection.return_value = ["item1"]

    gptscan.view_details(item_id="item1")

    # Find the Copy Code menu item and command
    assert "menu_Copy Code" in captured
    copy_code_cmd = captured["menu_Copy Code"]

    # Execute the command
    from gptscan import root as mock_root
    copy_code_cmd()

    # Verify clipboard and status bar feedback
    mock_root.clipboard_clear.assert_called()
    mock_root.clipboard_append.assert_called_with("print('hello')")
    # Verified feedback via status bar (it's the first label created in view_details)
    status_bar = captured['labels'][0]
    assert status_bar.config_data.get('text') == "Code copied to clipboard."

def test_view_details_copy_path_moved(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "test.py")

    # Verify button exists with new text
    assert "btn_Copy" in captured
    btn_mock, copy_path_cmd = captured["btn_Copy"]

    from gptscan import root as mock_root
    copy_path_cmd()
    mock_root.clipboard_append.assert_called_with("test.py")

def test_view_details_shortcuts(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "test.py", snippet="snippet_content")

    captured_bindings = {}
    mock_toplevel.bind.side_effect = lambda event, func: captured_bindings.update({event: func})

    gptscan.view_details(item_id="item1")

    assert '<Control-s>' in captured_bindings
    assert '<Command-s>' in captured_bindings

    from gptscan import root as mock_root
    captured_bindings['<Control-s>'](None)
    mock_root.clipboard_append.assert_called_with("snippet_content")

def test_view_details_keyboard_navigation_prevented(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    raw1 = ["file1.py", "10%", "", "", "", "snippet1", 1]
    mock_tree._item_values["item1"] = ["file1.py", "10%", "", "", "", "snippet1", 1, json.dumps(raw1)]
    raw2 = ["file2.py", "20%", "Admin", "User", "90%", "snippet2", 1]
    mock_tree._item_values["item2"] = ["file2.py", "20%", "Admin", "User", "90%", "snippet2", 1, json.dumps(raw2)]
    mock_tree.get_children.return_value = ["item1", "item2"]

    captured_bindings = {}
    mock_toplevel.bind.side_effect = lambda event, func: captured_bindings.update({event: func})

    # First, mock focus_get to return a mock widget with class "Text"
    mock_focused = MagicMock()
    mock_focused.winfo_class.return_value = "Text"
    mock_toplevel.focus_get.return_value = mock_focused

    gptscan.view_details(item_id="item1")
    assert '<Left>' in captured_bindings
    assert '<Right>' in captured_bindings

    # Trigger Right Key Press
    captured_bindings['<Right>'](None)
    # The selection should NOT have changed because focus was in a Text widget!
    mock_tree.selection_set.assert_not_called()

    # Now mock focus_get to return None, so navigation works
    mock_toplevel.focus_get.return_value = None

    # Reset mock to verify it works when not focused on Text/Entry
    mock_tree.selection_set.reset_mock()
    captured_bindings['<Right>'](None)
    mock_tree.selection_set.assert_called_with("item2")


def test_view_details_zoom_controls(mock_view_details_env, monkeypatch):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "item1", "test.py", snippet="snippet_content")

    # Find the zoom buttons
    assert "btn_+" in captured
    assert "btn_-" in captured
    assert "btn_100%" in captured

    # Mock the Font class cget/configure to track size changes
    mock_font_instance = MagicMock()
    mock_font_instance.cget.return_value = 10
    monkeypatch.setattr(gptscan.tkinter.font, 'Font', MagicMock(return_value=mock_font_instance))

    # Re-run setup_details to use the mocked Font class
    setup_details(mock_view_details_env, "item1", "test.py", snippet="snippet_content")

    # Capture the correct status bar (which is initialized to "Ready")
    status_bar = [l for l in captured['labels'] if l.config_data.get('text') == "Ready"][-1]

    # Click Zoom In
    zoom_in_btn, zoom_in_cmd = captured["btn_+"]
    zoom_in_cmd()
    mock_font_instance.configure.assert_called_with(size=11)
    assert "Font size set to 11pt." in status_bar.config_data.get('text', '')

    # Click Zoom Out
    zoom_out_btn, zoom_out_cmd = captured["btn_-"]
    zoom_out_cmd()
    mock_font_instance.configure.assert_called_with(size=10)
    assert "Font size set to 10pt." in status_bar.config_data.get('text', '')

    # Click Reset Zoom
    zoom_reset_btn, zoom_reset_cmd = captured["btn_100%"]
    zoom_reset_cmd()
    mock_font_instance.configure.assert_called_with(size=10)
    assert "Font size set to 10pt." in status_bar.config_data.get('text', '')


def test_view_details_zoom_shortcuts_and_mousewheel(mock_view_details_env, monkeypatch):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env

    # Track top level bindings
    captured_bindings = {}
    mock_toplevel.bind.side_effect = lambda event, func: captured_bindings.update({event: func})

    mock_font_instance = MagicMock()
    mock_font_instance.cget.return_value = 12
    monkeypatch.setattr(gptscan.tkinter.font, 'Font', MagicMock(return_value=mock_font_instance))

    setup_details(mock_view_details_env, "item1", "test.py", snippet="snippet_content")

    # Test keyboard shortcuts
    assert '<Control-plus>' in captured_bindings
    assert '<Control-minus>' in captured_bindings
    assert '<Control-Key-0>' in captured_bindings

    # Trigger Control-plus (Zoom In)
    captured_bindings['<Control-plus>'](None)
    mock_font_instance.configure.assert_any_call(size=13)

    # Trigger Control-minus (Zoom Out)
    captured_bindings['<Control-minus>'](None)
    mock_font_instance.configure.assert_any_call(size=12)

    # Trigger Control-0 (Reset)
    captured_bindings['<Control-Key-0>'](None)
    mock_font_instance.configure.assert_any_call(size=12)

    # Find MouseWheel bindings on ScrolledText
    snippet_st = captured['scrolledtexts'][-1]
    assert '<MouseWheel>' in snippet_st.bindings

    # Trigger MouseWheel Zoom In (Control state bitmask is 4, delta > 0)
    mock_event_zoom_in = MagicMock()
    mock_event_zoom_in.state = 4
    mock_event_zoom_in.delta = 120
    snippet_st.bindings['<MouseWheel>'](mock_event_zoom_in)
    mock_font_instance.configure.assert_any_call(size=13)

    # Trigger MouseWheel Zoom Out (Control state bitmask is 4, delta < 0)
    mock_event_zoom_out = MagicMock()
    mock_event_zoom_out.state = 4
    mock_event_zoom_out.delta = -120
    snippet_st.bindings['<MouseWheel>'](mock_event_zoom_out)
    mock_font_instance.configure.assert_any_call(size=12)

    # Trigger MouseWheel without Control (should do nothing to font)
    mock_font_instance.configure.reset_mock()
    mock_event_no_ctrl = MagicMock()
    mock_event_no_ctrl.state = 0
    mock_event_no_ctrl.delta = 120
    snippet_st.bindings['<MouseWheel>'](mock_event_no_ctrl)
    mock_font_instance.configure.assert_not_called()
