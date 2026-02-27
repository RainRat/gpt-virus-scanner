import sys
from unittest.mock import MagicMock, patch
import pytest
import gptscan

def test_auto_select_best_result_prioritizes_risks(monkeypatch):
    # Setup mocks
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    # Mock items: item1 is safe, item2 is high-risk, item3 is medium-risk
    items = ['item1', 'item2', 'item3']
    mock_tree.get_children.return_value = items

    # Mock tags for each item
    def mock_item_tags(iid, option):
        if iid == 'item1':
            return []
        elif iid == 'item2':
            return ['high-risk']
        elif iid == 'item3':
            return ['medium-risk']
        return []

    mock_tree.item.side_effect = mock_item_tags

    # Action
    gptscan._auto_select_best_result()

    # Assert: item2 should be selected as it's the first risk
    mock_tree.selection_set.assert_called_once_with('item2')
    mock_tree.focus.assert_called_once_with('item2')
    mock_tree.see.assert_called_once_with('item2')
    mock_tree.focus_set.assert_called_once()

def test_auto_select_best_result_falls_back_to_first_item(monkeypatch):
    # Setup mocks
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    # Mock items: all are safe
    items = ['item1', 'item2']
    mock_tree.get_children.return_value = items
    mock_tree.item.return_value = [] # No tags

    # Action
    gptscan._auto_select_best_result()

    # Assert: item1 should be selected
    mock_tree.selection_set.assert_called_once_with('item1')
    mock_tree.focus.assert_called_once_with('item1')
    mock_tree.see.assert_called_once_with('item1')

def test_auto_select_best_result_handles_empty_tree(monkeypatch):
    # Setup mocks
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    mock_tree.get_children.return_value = []

    # Action
    gptscan._auto_select_best_result()

    # Assert
    mock_tree.selection_set.assert_not_called()

def test_auto_select_best_result_handles_none_tree(monkeypatch):
    monkeypatch.setattr(gptscan, 'tree', None)
    # Should not raise
    gptscan._auto_select_best_result()
