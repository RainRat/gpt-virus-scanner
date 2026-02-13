import pytest
from unittest.mock import MagicMock, patch
import gptscan
import tkinter as tk

def test_matches_filter_basic(monkeypatch):
    """Test that _matches_filter correctly identifies matches."""
    mock_filter_var = MagicMock()
    monkeypatch.setattr(gptscan, 'filter_var', mock_filter_var)

    # Test case: exact match
    mock_filter_var.get.return_value = "suspicious"
    values = ("path/to/file.py", "90%", "Found suspicious call", "Malicious code", "85%", "eval(input())")
    assert gptscan._matches_filter(values) is True

    # Test case: case insensitive
    mock_filter_var.get.return_value = "PATH"
    assert gptscan._matches_filter(values) is True

    # Test case: no match
    mock_filter_var.get.return_value = "clean"
    assert gptscan._matches_filter(values) is False

    # Test case: empty filter
    mock_filter_var.get.return_value = ""
    assert gptscan._matches_filter(values) is True

def test_apply_filter_refreshes_tree(monkeypatch):
    """Test that _apply_filter clears and repopulates the tree."""
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    mock_filter_var = MagicMock()
    mock_filter_var.get.return_value = "test"
    monkeypatch.setattr(gptscan, 'filter_var', mock_filter_var)

    # Setup cache
    results = [
        ("test.py", "90%", "Admin", "User", "80%", "snippet"),
        ("safe.py", "10%", "Admin", "User", "0%", "snippet")
    ]
    monkeypatch.setattr(gptscan, '_all_results_cache', results)

    # Mock _prepare_tree_row
    monkeypatch.setattr(gptscan, '_prepare_tree_row', lambda v: (list(v), ()))

    # Run filter
    gptscan._apply_filter()

    # Verify tree was cleared
    mock_tree.get_children.assert_called_once()
    mock_tree.delete.assert_called_once()

    # Verify only matching item was inserted
    # It should be called once for "test.py"
    assert mock_tree.insert.call_count == 1
    _, kwargs = mock_tree.insert.call_args
    assert kwargs['values'][0] == "test.py"

def test_insert_tree_row_updates_cache(monkeypatch):
    """Test that insert_tree_row appends to cache."""
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    monkeypatch.setattr(gptscan, '_all_results_cache', [])
    monkeypatch.setattr(gptscan, 'filter_var', None) # No filter
    monkeypatch.setattr(gptscan, '_prepare_tree_row', lambda v: (list(v), ()))

    values = ("new.py", "50%", "Admin", "User", "", "Snippet")
    gptscan.insert_tree_row(values)

    assert len(gptscan._all_results_cache) == 1
    assert gptscan._all_results_cache[0] == values
    assert mock_tree.insert.called

def test_clear_results_clears_cache(monkeypatch):
    """Test that clear_results empties the cache."""
    monkeypatch.setattr(gptscan, '_all_results_cache', [("some", "data")])
    monkeypatch.setattr(gptscan, 'tree', MagicMock())
    monkeypatch.setattr(gptscan, 'progress_bar', None)
    monkeypatch.setattr(gptscan, 'status_label', None)

    gptscan.clear_results()
    assert len(gptscan._all_results_cache) == 0

def test_select_all_items(monkeypatch):
    """Test that select_all_items selects all tree children."""
    mock_tree = MagicMock()
    mock_tree.get_children.return_value = ("item1", "item2")
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    result = gptscan.select_all_items()

    mock_tree.selection_set.assert_called_with(("item1", "item2"))
    assert result == "break"
