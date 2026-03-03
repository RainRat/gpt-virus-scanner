import json
import pytest
from unittest.mock import MagicMock
import gptscan

@pytest.fixture
def mock_tree(monkeypatch):
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    return mock_tree

def test_get_item_raw_values_from_cache(mock_tree):
    # Mock item values with valid JSON in index 7
    raw_data = ["path/to/file", "90%", "Admin Notes", "User Notes", "85%", "print('hello')", "10"]
    item_values = ["path/to/file", "90%", "Admin Notes", "User Notes", "85%", "print('hello')", "10", json.dumps(raw_data)]

    mock_tree.exists.return_value = True
    # mock_tree.item(iid, "values") returns the list of values directly
    mock_tree.item.return_value = item_values

    result = gptscan._get_item_raw_values("item1")

    assert result == raw_data
    mock_tree.exists.assert_called_once_with("item1")
    mock_tree.item.assert_called_once_with("item1", "values")

def test_get_item_raw_values_fallback(mock_tree):
    # Mock item values without cache (or too short)
    item_values = ["path/to/file", "90%", "Admin Notes", "User Notes", "85%", "snippet\nwith\nnewline", "10"]

    mock_tree.exists.return_value = True
    mock_tree.item.return_value = item_values

    result = gptscan._get_item_raw_values("item1")

    # Fallback should replace newlines with spaces and take first 7 elements
    expected = ["path/to/file", "90%", "Admin Notes", "User Notes", "85%", "snippet with newline", "10"]
    assert result == expected

def test_get_item_raw_values_invalid_cache(mock_tree):
    # Mock item values with invalid JSON in index 7
    item_values = ["p", "o", "a", "u", "g", "s", "l", "invalid-json"]

    mock_tree.exists.return_value = True
    mock_tree.item.return_value = item_values

    result = gptscan._get_item_raw_values("item1")

    # Should fallback to basic parsing
    assert result == ["p", "o", "a", "u", "g", "s", "l"]

def test_get_item_raw_values_none_tree(monkeypatch):
    monkeypatch.setattr(gptscan, 'tree', None)
    assert gptscan._get_item_raw_values("item1") is None

def test_get_item_raw_values_item_not_exists(mock_tree):
    mock_tree.exists.return_value = False
    assert gptscan._get_item_raw_values("non-existent") is None

def test_get_item_raw_values_empty_values(mock_tree):
    mock_tree.exists.return_value = True
    mock_tree.item.return_value = []

    result = gptscan._get_item_raw_values("item1")
    assert result == []

def test_get_selected_row_values(mock_tree, monkeypatch):
    raw_data = ["path", "conf"]
    # Mock _get_item_raw_values to return our data
    monkeypatch.setattr(gptscan, "_get_item_raw_values", lambda iid: raw_data if iid == "sel1" else None)

    mock_tree.selection.return_value = ("sel1",)

    assert gptscan._get_selected_row_values() == raw_data

    # Test no selection
    mock_tree.selection.return_value = ()
    assert gptscan._get_selected_row_values() is None

def test_get_selected_row_values_no_tree(monkeypatch):
    monkeypatch.setattr(gptscan, 'tree', None)
    assert gptscan._get_selected_row_values() is None
