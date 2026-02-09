import sys
from unittest.mock import MagicMock
import pytest
import gptscan

def test_update_tree_columns_checked(monkeypatch):
    # Setup mocks
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)

    mock_gpt_var = MagicMock()
    mock_gpt_var.get.return_value = True
    monkeypatch.setattr(gptscan, 'gpt_var', mock_gpt_var, raising=False)

    # Action
    gptscan.update_tree_columns()

    # Assert - should show all columns
    all_cols = ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet")
    mock_tree.__setitem__.assert_called_with("displaycolumns", all_cols)

def test_update_tree_columns_unchecked_empty(monkeypatch):
    # Setup mocks
    mock_tree = MagicMock()
    mock_tree.get_children.return_value = []
    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)

    mock_gpt_var = MagicMock()
    mock_gpt_var.get.return_value = False
    monkeypatch.setattr(gptscan, 'gpt_var', mock_gpt_var, raising=False)

    # Action
    gptscan.update_tree_columns()

    # Assert - should show local columns only
    local_cols = ("path", "own_conf", "snippet")
    mock_tree.__setitem__.assert_called_with("displaycolumns", local_cols)

def test_update_tree_columns_unchecked_with_ai_data(monkeypatch):
    # Setup mocks
    mock_tree = MagicMock()
    mock_tree.get_children.return_value = ["item1"]
    # Mock tree.set(iid, "gpt_conf") to return a value
    mock_tree.set.return_value = "90%"
    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)

    mock_gpt_var = MagicMock()
    mock_gpt_var.get.return_value = False
    monkeypatch.setattr(gptscan, 'gpt_var', mock_gpt_var, raising=False)

    # Action
    gptscan.update_tree_columns()

    # Assert - should show all columns because of AI data
    all_cols = ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet")
    mock_tree.__setitem__.assert_called_with("displaycolumns", all_cols)
    mock_tree.set.assert_called_with("item1", "gpt_conf")

def test_insert_tree_row_triggers_update(monkeypatch):
    # Setup
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)

    mock_update = MagicMock()
    monkeypatch.setattr(gptscan, 'update_tree_columns', mock_update)

    # Mock other things insert_tree_row needs
    monkeypatch.setattr(gptscan, 'default_font_measure', lambda x: 10)
    mock_font = MagicMock()
    mock_font.measure.return_value = 10
    monkeypatch.setattr(gptscan.tkinter.font, 'Font', lambda **k: mock_font)

    mock_tree.column.return_value = {'width': 100}
    mock_tree["columns"] = ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet")

    # Action
    values = ("test.py", "50%", "admin", "user", "90%", "code")
    gptscan.insert_tree_row(values)

    # Assert
    mock_update.assert_called_once()

def test_button_click_triggers_update(monkeypatch):
    # Setup mocks for button_click
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)

    mock_update = MagicMock()
    monkeypatch.setattr(gptscan, 'update_tree_columns', mock_update)

    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "" # To stop early
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox, raising=False)

    monkeypatch.setattr(gptscan, 'current_cancel_event', None)
    monkeypatch.setattr(gptscan, 'messagebox', MagicMock())

    # Action
    gptscan.button_click()

    # Assert
    mock_update.assert_called_once()
    mock_tree.delete.assert_called_once()
