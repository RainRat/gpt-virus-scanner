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

def test_on_filter_escape_clears_and_focuses(monkeypatch):
    """Test that on_filter_escape clears the filter, applies the filter, and refocuses tree."""
    mock_filter_var = MagicMock()
    mock_tree = MagicMock()

    monkeypatch.setattr(gptscan, 'filter_var', mock_filter_var)
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    mock_apply_filter = MagicMock()
    mock_on_filter_return = MagicMock()

    monkeypatch.setattr(gptscan, '_apply_filter', mock_apply_filter)
    monkeypatch.setattr(gptscan, 'on_filter_return', mock_on_filter_return)

    # Mock necessary GUI config globals to avoid warnings
    monkeypatch.setattr(gptscan.Config, 'recent_paths', [])
    monkeypatch.setattr(gptscan.Config, 'last_path', "")
    monkeypatch.setattr(gptscan.Config, 'extensions_missing', False)

    # Mock select_range and focus_set on the Combobox mock class (MockWidget)
    monkeypatch.setattr(gptscan.tk.ttk.Combobox, 'select_range', lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(gptscan.tk.ttk.Combobox, 'focus_set', lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(gptscan.tk.ttk.Combobox, 'get', lambda *a, **kw: "openai", raising=False)

    # Mock after on root (MockTk)
    monkeypatch.setattr(gptscan.tk.Tk, 'after', lambda *a, **kw: None, raising=False)

    # Mock Button and LabelFrame on ttk to avoid MagicMock spec errors
    monkeypatch.setattr(gptscan.tk.ttk, 'Button', gptscan.tk.ttk.Frame, raising=False)
    monkeypatch.setattr(gptscan.tk.ttk, 'LabelFrame', gptscan.tk.ttk.Frame, raising=False)
    monkeypatch.setattr(gptscan.tk, 'Button', gptscan.tk.ttk.Frame, raising=False)
    monkeypatch.setattr(gptscan.tk, 'LabelFrame', gptscan.tk.ttk.Frame, raising=False)

    # Track the filter entry and other entries created in create_gui
    mock_created_entry = MagicMock()
    monkeypatch.setattr(gptscan.tk.ttk, 'Entry', lambda *args, **kwargs: mock_created_entry, raising=False)
    monkeypatch.setattr(gptscan.tk, 'Entry', lambda *args, **kwargs: mock_created_entry, raising=False)
    monkeypatch.setattr(gptscan.tk.ttk, 'Treeview', lambda *args, **kwargs: MagicMock(), raising=False)

    with patch('gptscan.messagebox.showwarning'):
        gptscan.create_gui()

    # Now find the binding function registered to '<Escape>'
    escape_binding = None
    for call in mock_created_entry.bind.call_args_list:
        args, kwargs = call
        if args and args[0] == '<Escape>':
            escape_binding = args[1]
            break

    assert escape_binding is not None, "Escape key binding was not registered on filter_entry"

    # Invoke the registered escape binding
    res = escape_binding(MagicMock())

    # Assertions
    gptscan.filter_var.set.assert_called_with("")
    mock_apply_filter.assert_called_once()
    mock_on_filter_return.assert_called_once()
    assert res == "break"

    # Cleanup mutated global variables to prevent cross-test state leakage
    gptscan.root = None
    gptscan.textbox = None
    gptscan.progress_bar = None
    gptscan.status_label = None
    gptscan.deep_var = None
    gptscan.all_var = None
    gptscan.scan_all_var = None
    gptscan.gpt_var = None
    gptscan.dry_var = None
    gptscan.git_var = None
    gptscan.filter_var = None
    gptscan.filter_entry = None
    gptscan.tree = None
    gptscan.scan_button = None
    gptscan.view_button = None
    gptscan.intel_button = None
    gptscan.intel_menu = None
    gptscan.rescan_button = None
    gptscan.open_button = None
    gptscan.analyze_button = None
    gptscan.exclude_button = None
    gptscan.reveal_button = None
    gptscan.results_button = None
    gptscan.browse_button = None
    gptscan.show_key_btn = None
    gptscan.default_font_measure = None
    gptscan.copy_cmd_button = None
    gptscan.clear_target_btn = None
    gptscan.git_checkbox = None
    gptscan.deep_checkbox = None
    gptscan.scan_all_checkbox = None
    gptscan.dry_checkbox = None
    gptscan.gpt_checkbox = None
    gptscan.provider_combo = None
    gptscan.model_combo = None
    gptscan.api_key_entry = None
    gptscan.api_entry = None
    gptscan.all_checkbox = None
    gptscan.threshold_spin = None
    gptscan.provider_var = None
    gptscan.model_var = None
    gptscan.api_base_var = None
    gptscan.api_key_var = None
