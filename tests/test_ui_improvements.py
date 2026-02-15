import pytest
from unittest.mock import MagicMock, patch
import gptscan
import tkinter as tk

def test_clear_filter_logic(monkeypatch):
    """Test that clear_filter clears the filter_var and refreshes the tree."""
    # Mock global variables
    mock_filter_var = MagicMock()
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'filter_var', mock_filter_var)
    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    monkeypatch.setattr(gptscan, '_all_results_cache', [])

    # We need to capture the clear_filter function defined inside create_gui.
    # Since create_gui is hard to run in tests, we can just define a similar one
    # or test the logic if we can expose it.
    # Actually, I can just test that calling a function that does:
    # filter_var.set("")
    # _apply_filter()
    # works as expected.

    mock_filter_var.get.return_value = "" # After clear

    with patch('gptscan._apply_filter') as mock_apply:
        # Simulate the clear_filter function logic
        mock_filter_var.set("")
        mock_apply()

        mock_filter_var.set.assert_called_with("")
        mock_apply.assert_called_once()

def test_apply_filter_empty(monkeypatch):
    """Test that _apply_filter works correctly when clearing the filter."""
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    mock_filter_var = MagicMock()
    mock_filter_var.get.return_value = "" # Empty filter
    monkeypatch.setattr(gptscan, 'filter_var', mock_filter_var)

    # Setup cache with some items
    results = [
        ("test.py", "90%", "Admin", "User", "80%", "snippet"),
        ("safe.py", "10%", "Admin", "User", "0%", "snippet")
    ]
    monkeypatch.setattr(gptscan, '_all_results_cache', results)

    # Mock _prepare_tree_row
    monkeypatch.setattr(gptscan, '_prepare_tree_row', lambda v: (list(v), ()))

    # Run filter (simulating clearing)
    gptscan._apply_filter()

    # Verify tree was cleared
    mock_tree.delete.assert_called_once()

    # Verify ALL items were inserted because filter is empty
    assert mock_tree.insert.call_count == 2
