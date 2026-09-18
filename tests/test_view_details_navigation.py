import json
import tkinter as tk
from unittest.mock import MagicMock
import pytest
import gptscan


@pytest.fixture
def mock_details_navigation_env(monkeypatch):
    """Setup a mock environment specifically for view_details navigation tests."""
    mock_tree = MagicMock()
    mock_tree.__getitem__.side_effect = lambda key: (
        "path",
        "own_conf",
        "admin_desc",
        "end-user_desc",
        "gpt_conf",
        "snippet",
        "line",
    ) if key == "columns" else MagicMock()
    monkeypatch.setattr(gptscan, "tree", mock_tree)

    mock_tree._item_values = {}

    def mock_item_func(item_id, option=None):
        vals = mock_tree._item_values.get(item_id, [])
        if option == "values":
            return vals
        return {"values": vals}

    mock_tree.item.side_effect = mock_item_func
    mock_tree.exists.side_effect = lambda iid: iid in mock_tree._item_values

    mock_root = MagicMock()
    monkeypatch.setattr(gptscan, "root", mock_root)

    mock_toplevel = MagicMock()
    monkeypatch.setattr(gptscan.tk, "Toplevel", MagicMock(return_value=mock_toplevel))

    captured_bindings = {}

    def mock_bind(event, func=None, add=None):
        if func is not None:
            captured_bindings[event] = func

    mock_toplevel.bind.side_effect = mock_bind

    captured_buttons = {}

    def mock_button_init(master, **kwargs):
        btn = MagicMock()
        text = kwargs.get("text", "")
        if text:
            captured_buttons[text] = (btn, kwargs.get("command"))
        return btn

    monkeypatch.setattr(gptscan.ttk, "Button", mock_button_init)

    class MockLabel:
        def __init__(self, *args, **kwargs):
            pass

        def config(self, **kwargs):
            pass

        def cget(self, key):
            return ""

        def grid(self, **kwargs):
            pass

        def pack(self, **kwargs):
            pass

        def grid_forget(self):
            pass

        def pack_forget(self):
            pass

        def winfo_viewable(self):
            return True

    monkeypatch.setattr(gptscan.tk, "Label", MockLabel)
    monkeypatch.setattr(gptscan.ttk, "Label", MockLabel)

    class MockScrolledText:
        def __init__(self, *args, **kwargs):
            self.content = ""

        def bind(self, sequence, func=None, add=None):
            pass

        def delete(self, start, end):
            self.content = ""

        def insert(self, idx, val):
            self.content += val

        def get(self, start, end):
            return self.content

        def config(self, **kwargs):
            pass

        def pack(self, **kwargs):
            pass

        def pack_forget(self):
            pass

        def see(self, *args):
            pass

        def tag_configure(self, *args, **kwargs):
            pass

        def tag_add(self, *args, **kwargs):
            pass

        def winfo_viewable(self):
            return True

    monkeypatch.setattr(gptscan.scrolledtext, "ScrolledText", MockScrolledText)

    return mock_tree, mock_toplevel, captured_bindings, captured_buttons


def _populate_tree_items(mock_tree, item_ids):
    for iid in item_ids:
        raw = [f"file_{iid}.py", "10%", "", "", "", f"snippet_{iid}", 1]
        mock_tree._item_values[iid] = [
            f"file_{iid}.py",
            "10%",
            "",
            "",
            "",
            f"snippet_{iid}",
            1,
            json.dumps(raw),
        ]
    mock_tree.get_children.return_value = list(item_ids)


def test_is_input_focused_scenarios(mock_details_navigation_env):
    mock_tree, mock_toplevel, captured_bindings, captured_buttons = (
        mock_details_navigation_env
    )
    _populate_tree_items(mock_tree, ["item1"])

    mock_focus_widget = MagicMock()
    mock_toplevel.focus_get.return_value = mock_focus_widget

    gptscan.view_details(item_id="item1")

    left_handler = captured_bindings["<Left>"]

    # 1. Focused on an Entry widget
    mock_focus_widget.winfo_class.return_value = "Entry"
    mock_tree.selection_set.reset_mock()
    left_handler(None)
    mock_tree.selection_set.assert_not_called()

    # 2. Focused on Text widget
    mock_focus_widget.winfo_class.return_value = "Text"
    left_handler(None)
    mock_tree.selection_set.assert_not_called()

    # 3. Focused on TEntry widget
    mock_focus_widget.winfo_class.return_value = "TEntry"
    left_handler(None)
    mock_tree.selection_set.assert_not_called()

    # 4. Focused on non-input widget (e.g. Frame or Label)
    mock_focus_widget.winfo_class.return_value = "Frame"
    left_handler(None)
    # Selection set is called when focused on non-input widget and navigating
    # But since item1 is the only item, selection_set won't be called for prev/next boundary
    # Let's test with 2 items below.

    # 5. focus_get() returns None
    mock_toplevel.focus_get.return_value = None
    left_handler(None)
    mock_tree.selection_set.assert_not_called()

    # 6. focus_get() raises Exception
    mock_focus_widget.winfo_class.side_effect = Exception("Widget error")
    mock_toplevel.focus_get.return_value = mock_focus_widget
    left_handler(None)
    mock_tree.selection_set.assert_not_called()


def test_navigate_prev_and_next_flow(mock_details_navigation_env):
    mock_tree, mock_toplevel, captured_bindings, captured_buttons = (
        mock_details_navigation_env
    )
    _populate_tree_items(mock_tree, ["item1", "item2", "item3"])
    mock_toplevel.focus_get.return_value = None

    gptscan.view_details(item_id="item2")

    prev_btn, on_prev_cmd = captured_buttons["< Previous"]
    next_btn, on_next_cmd = captured_buttons["Next >"]

    # Navigate Next: item2 -> item3
    on_next_cmd()
    mock_tree.selection_set.assert_called_with("item3")

    # Navigate Next at last item: item3 -> boundary (should do nothing)
    mock_tree.selection_set.reset_mock()
    on_next_cmd()
    mock_tree.selection_set.assert_not_called()

    # Navigate Prev: item3 -> item2
    on_prev_cmd()
    mock_tree.selection_set.assert_called_with("item2")

    # Navigate Prev: item2 -> item1
    on_prev_cmd()
    mock_tree.selection_set.assert_called_with("item1")

    # Navigate Prev at first item: item1 -> boundary (should do nothing)
    mock_tree.selection_set.reset_mock()
    on_prev_cmd()
    mock_tree.selection_set.assert_not_called()


def test_navigate_shortcuts_unfocused(mock_details_navigation_env):
    mock_tree, mock_toplevel, captured_bindings, captured_buttons = (
        mock_details_navigation_env
    )
    _populate_tree_items(mock_tree, ["item1", "item2"])
    mock_toplevel.focus_get.return_value = None

    gptscan.view_details(item_id="item1")

    # Alt+Right shortcut direct call
    alt_right_handler = captured_bindings["<Alt-Right>"]
    alt_right_handler(None)
    mock_tree.selection_set.assert_called_with("item2")

    # Alt+Left shortcut direct call
    alt_left_handler = captured_bindings["<Alt-Left>"]
    alt_left_handler(None)
    mock_tree.selection_set.assert_called_with("item1")


def test_navigate_value_error_fallback(mock_details_navigation_env):
    mock_tree, mock_toplevel, captured_bindings, captured_buttons = (
        mock_details_navigation_env
    )
    _populate_tree_items(mock_tree, ["item1", "item2"])
    mock_toplevel.focus_get.return_value = None

    gptscan.view_details(item_id="item1")

    # Simulate item removed from tree visible items
    mock_tree.get_children.return_value = ["item2", "item3"]

    alt_right_handler = captured_bindings["<Alt-Right>"]
    mock_tree.selection_set.reset_mock()
    # current_item_id is item1, which is not in get_children(), causing ValueError which should be caught
    alt_right_handler(None)
    mock_tree.selection_set.assert_not_called()
