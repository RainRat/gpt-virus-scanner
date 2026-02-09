import tkinter as tk
from unittest.mock import MagicMock
import pytest
import gptscan

def test_update_tree_columns_hides_ai_when_disabled_and_no_data(monkeypatch):
    # Setup mocks
    mock_tree = MagicMock()
    mock_tree.__getitem__.return_value = ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet")
    mock_tree.get_children.return_value = []

    mock_gpt_var = MagicMock()
    mock_gpt_var.get.return_value = False

    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    monkeypatch.setattr(gptscan, 'gpt_var', mock_gpt_var)

    # Run
    gptscan.update_tree_columns()

    # Verify
    expected_cols = ("path", "own_conf", "snippet")
    mock_tree.__setitem__.assert_called_with("displaycolumns", expected_cols)

def test_update_tree_columns_shows_ai_when_enabled(monkeypatch):
    # Setup mocks
    mock_tree = MagicMock()
    mock_tree.get_children.return_value = []

    mock_gpt_var = MagicMock()
    mock_gpt_var.get.return_value = True

    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    monkeypatch.setattr(gptscan, 'gpt_var', mock_gpt_var)

    # Run
    gptscan.update_tree_columns()

    # Verify
    expected_cols = ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet")
    mock_tree.__setitem__.assert_called_with("displaycolumns", expected_cols)

def test_update_tree_columns_shows_ai_when_data_present_even_if_disabled(monkeypatch):
    # Setup mocks
    mock_tree = MagicMock()

    # Mock some children data
    item_id = "item1"
    mock_tree.get_children.return_value = [item_id]
    # values[4] is gpt_conf. We set it to something non-empty.
    # The code calls tree.item(item_id, 'values')
    mock_tree.item.return_value = ("file.py", "10%", "Admin says bad", "User avoid", "90%", "snippet")

    mock_gpt_var = MagicMock()
    mock_gpt_var.get.return_value = False

    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    monkeypatch.setattr(gptscan, 'gpt_var', mock_gpt_var)

    # Run
    gptscan.update_tree_columns()

    # Verify
    expected_cols = ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet")
    mock_tree.__setitem__.assert_called_with("displaycolumns", expected_cols)

def test_update_tree_columns_hides_ai_when_disabled_and_data_is_empty(monkeypatch):
    # Setup mocks
    mock_tree = MagicMock()

    item_id = "item1"
    mock_tree.get_children.return_value = [item_id]
    # values[4] is gpt_conf. We set it to empty string.
    mock_tree.item.return_value = ("file.py", "10%", "", "", "", "snippet")

    mock_gpt_var = MagicMock()
    mock_gpt_var.get.return_value = False

    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    monkeypatch.setattr(gptscan, 'gpt_var', mock_gpt_var)

    # Run
    gptscan.update_tree_columns()

    # Verify
    expected_cols = ("path", "own_conf", "snippet")
    mock_tree.__setitem__.assert_called_with("displaycolumns", expected_cols)
