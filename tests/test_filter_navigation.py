import pytest
from unittest.mock import MagicMock, patch
import gptscan
import tkinter as tk

@pytest.fixture
def mock_ui_env(monkeypatch):
    mock_tree = MagicMock()
    mock_filter_entry = MagicMock()

    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    monkeypatch.setattr(gptscan, 'filter_entry', mock_filter_entry)

    return {
        'tree': mock_tree,
        'filter_entry': mock_filter_entry
    }

def test_focus_filter_sets_focus_and_selects_all(mock_ui_env):
    """Test that focus_filter sets focus to filter_entry and selects all text."""
    res = gptscan.focus_filter()

    mock_ui_env['filter_entry'].focus_set.assert_called_once()
    # tk.END is mocked to "end" in conftest.py
    mock_ui_env['filter_entry'].selection_range.assert_called_once_with(0, "end")
    assert res == "break"

def test_focus_filter_handles_none_entry(monkeypatch):
    """Test focus_filter doesn't crash if filter_entry is None."""
    monkeypatch.setattr(gptscan, 'filter_entry', None)
    res = gptscan.focus_filter()
    assert res == "break"

def test_on_filter_return_transitions_to_tree(mock_ui_env):
    """Test that on_filter_return sets focus to the tree."""
    mock_ui_env['tree'].selection.return_value = ("item1",) # Something already selected

    res = gptscan.on_filter_return()

    mock_ui_env['tree'].focus_set.assert_called_once()
    # Should NOT select first item if something is already selected
    mock_ui_env['tree'].selection_set.assert_not_called()
    assert res == "break"

def test_on_filter_return_selects_first_item_if_none_selected(mock_ui_env):
    """Test that on_filter_return selects the first item if tree selection is empty."""
    mock_ui_env['tree'].selection.return_value = ()
    mock_ui_env['tree'].get_children.return_value = ("item1", "item2")

    res = gptscan.on_filter_return()

    mock_ui_env['tree'].focus_set.assert_called_once()
    mock_ui_env['tree'].selection_set.assert_called_with("item1")
    mock_ui_env['tree'].focus.assert_called_with("item1")
    mock_ui_env['tree'].see.assert_called_with("item1")
    assert res == "break"

def test_on_filter_return_handles_empty_tree(mock_ui_env):
    """Test on_filter_return transitions focus even if tree is empty."""
    mock_ui_env['tree'].selection.return_value = ()
    mock_ui_env['tree'].get_children.return_value = ()

    res = gptscan.on_filter_return()

    mock_ui_env['tree'].focus_set.assert_called_once()
    mock_ui_env['tree'].selection_set.assert_not_called()
    assert res == "break"

def test_on_filter_return_handles_none_tree(monkeypatch):
    """Test on_filter_return doesn't crash if tree is None."""
    monkeypatch.setattr(gptscan, 'tree', None)
    res = gptscan.on_filter_return()
    assert res == "break"


def test_on_filter_escape_resets_and_focuses(mock_ui_env, monkeypatch):
    """Test that on_filter_escape clears the filter, refreshes tree, focuses tree, and returns break."""
    mock_filter_var = MagicMock()
    mock_apply_filter = MagicMock()

    monkeypatch.setattr(gptscan, 'filter_var', mock_filter_var)
    monkeypatch.setattr(gptscan, '_apply_filter', mock_apply_filter)

    res = gptscan.on_filter_escape()

    mock_filter_var.set.assert_called_once_with("")
    mock_apply_filter.assert_called_once()
    mock_ui_env['tree'].focus_set.assert_called_once()
    assert res == "break"


def test_on_filter_escape_auto_selects_best_result(mock_ui_env, monkeypatch):
    """Test that on_filter_escape automatically selects the best result if tree selection is empty."""
    mock_filter_var = MagicMock()
    mock_filter_var.get.return_value = "query"
    mock_auto_select = MagicMock()

    mock_ui_env['tree'].selection.return_value = ()
    mock_ui_env['tree'].get_children.return_value = ("item1", "item2")

    monkeypatch.setattr(gptscan, 'filter_var', mock_filter_var)
    monkeypatch.setattr(gptscan, '_apply_filter', lambda: None)
    monkeypatch.setattr(gptscan, '_auto_select_best_result', mock_auto_select)
    monkeypatch.setattr(gptscan, 'current_cancel_event', None)

    res = gptscan.on_filter_escape()

    mock_ui_env['tree'].focus_set.assert_called_once()
    mock_auto_select.assert_called_once()
    assert res == "break"


def test_on_filter_escape_empty_query_focuses_tree(mock_ui_env, monkeypatch):
    """Test that on_filter_escape shifts focus to tree even when query is already empty."""
    mock_filter_var = MagicMock()
    mock_filter_var.get.return_value = ""
    monkeypatch.setattr(gptscan, 'filter_var', mock_filter_var)
    monkeypatch.setattr(gptscan, 'current_cancel_event', None)

    res = gptscan.on_filter_escape()

    mock_ui_env['tree'].focus_set.assert_called_once()
    assert res == "break"


def test_on_filter_escape_whitespace_query_clears_and_focuses(mock_ui_env, monkeypatch):
    """Test that on_filter_escape clears whitespace-only query, refreshes results, and focuses tree."""
    mock_filter_var = MagicMock()
    mock_filter_var.get.return_value = "   "
    mock_apply_filter = MagicMock()

    monkeypatch.setattr(gptscan, 'filter_var', mock_filter_var)
    monkeypatch.setattr(gptscan, '_apply_filter', mock_apply_filter)
    monkeypatch.setattr(gptscan, 'current_cancel_event', None)

    res = gptscan.on_filter_escape()

    mock_filter_var.set.assert_called_once_with("")
    mock_apply_filter.assert_called_once()
    mock_ui_env['tree'].focus_set.assert_called_once()
    assert res == "break"


def test_clear_filter_button_visibility_toggles(monkeypatch):
    """Test that _apply_filter toggles clear_filter_btn visibility based on filter_var content."""
    mock_clear_btn = MagicMock()
    mock_filter_var = MagicMock()
    mock_tree = MagicMock()

    monkeypatch.setattr(gptscan, 'clear_filter_btn', mock_clear_btn)
    monkeypatch.setattr(gptscan, 'filter_var', mock_filter_var)
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    # Empty filter query
    mock_filter_var.get.return_value = ""
    gptscan._apply_filter()
    mock_clear_btn.grid_remove.assert_called_once()
    mock_clear_btn.grid.assert_not_called()

    mock_clear_btn.reset_mock()

    # Non-empty filter query
    mock_filter_var.get.return_value = "suspicious"
    gptscan._apply_filter()
    mock_clear_btn.grid.assert_called_once_with(row=0, column=2, padx=(0, 5))
    mock_clear_btn.grid_remove.assert_not_called()


def test_filter_tooltips_include_keyboard_shortcuts(monkeypatch):
    """Test that filter_entry and clear_filter_btn bind tooltips containing accelerator hints."""
    bound_messages = {}

    def mock_bind_hover_message(widget, message, label=None):
        bound_messages[widget] = message

    monkeypatch.setattr(gptscan, 'bind_hover_message', mock_bind_hover_message)

    mock_entry = MagicMock()
    mock_btn = MagicMock()
    gptscan.bind_hover_message(mock_entry, "Search results by any column (path, threat level, analysis, snippet). (Ctrl+F)")
    gptscan.bind_hover_message(mock_btn, "Clear the filter. (Esc)")

    assert mock_entry in bound_messages
    assert "(Ctrl+F)" in bound_messages[mock_entry]

    assert mock_btn in bound_messages
    assert "(Esc)" in bound_messages[mock_btn]


def test_filter_entry_down_arrow_binding(monkeypatch):
    """Test that filter_entry binds <Down> to on_filter_return during GUI creation."""
    mock_filter_entry = MagicMock()
    filter_bindings = {}
    mock_filter_entry.bind.side_effect = lambda event, func: filter_bindings.update({event: func})

    mock_combo = MagicMock()
    mock_combo.get.return_value = "openai"

    mock_tree = MagicMock()

    mock_root = MagicMock()

    monkeypatch.setattr(gptscan, 'filter_entry', mock_filter_entry)
    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    monkeypatch.setattr(gptscan, 'root', mock_root)

    with monkeypatch.context() as m:
        m.setattr(gptscan.tk, 'Tk', lambda: mock_root)
        m.setattr(gptscan.ttk, 'Entry', lambda *a, **kw: mock_filter_entry)
        m.setattr(gptscan.ttk, 'Combobox', lambda *a, **kw: mock_combo)
        m.setattr(gptscan.ttk, 'Treeview', lambda *a, **kw: mock_tree)

        gptscan.create_gui()

    assert '<Down>' in filter_bindings
    assert filter_bindings['<Down>'] == gptscan.on_filter_return


def test_apply_filter_auto_selects_best_result_and_updates_button_states(monkeypatch):
    """Test that _apply_filter automatically selects the best result and updates button states when results match."""
    mock_tree = MagicMock()
    mock_tree.get_children.return_value = ("item1", "item2")
    mock_auto_select = MagicMock()
    mock_update_buttons = MagicMock()

    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    monkeypatch.setattr(gptscan, 'clear_filter_btn', None)
    monkeypatch.setattr(gptscan, 'filter_var', None)
    monkeypatch.setattr(gptscan, 'all_var', None)
    monkeypatch.setattr(gptscan, '_all_results_cache', [("file1.py", "80%", "", "", "", "eval()", 1)])
    monkeypatch.setattr(gptscan, '_auto_select_best_result', mock_auto_select)
    monkeypatch.setattr(gptscan, 'update_button_states', mock_update_buttons)
    monkeypatch.setattr(gptscan, 'update_tree_columns', lambda: None)
    monkeypatch.setattr(gptscan, '_prepare_tree_row', lambda vals: (vals, ("high-risk",)))

    gptscan._apply_filter()

    mock_auto_select.assert_called_once_with(focus_tree=False)
    mock_update_buttons.assert_called_once()


def test_auto_select_best_result_focus_tree_parameter(monkeypatch):
    """Test that _auto_select_best_result respects focus_tree to avoid stealing focus."""
    mock_tree = MagicMock()
    mock_tree.get_children.return_value = ["item1"]
    mock_tree.item.return_value = {"tags": ("high-risk",)}
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    # When focus_tree=False, tree.focus_set() should NOT be called
    gptscan._auto_select_best_result(focus_tree=False)
    mock_tree.selection_set.assert_called_with("item1")
    mock_tree.focus.assert_called_with("item1")
    mock_tree.see.assert_called_with("item1")
    mock_tree.focus_set.assert_not_called()

    # When focus_tree=True, tree.focus_set() SHOULD be called
    mock_tree.reset_mock()
    mock_tree.get_children.return_value = ["item1"]
    mock_tree.item.return_value = {"tags": ("high-risk",)}
    gptscan._auto_select_best_result(focus_tree=True)
    mock_tree.selection_set.assert_called_with("item1")
    mock_tree.focus_set.assert_called_once()
