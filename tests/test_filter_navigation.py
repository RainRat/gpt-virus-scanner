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
    # The mocked tk.END value is "end" (see tests/conftest.py)
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
