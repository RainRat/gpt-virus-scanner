import pytest
from unittest.mock import MagicMock, patch
import gptscan

@pytest.fixture
def mock_tree(monkeypatch):
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    return mock_tree

def test_update_tree_row_exists_matches(mock_tree, monkeypatch):
    original_values = ("path1.py", "80%", "", "", "", "snippet1")
    monkeypatch.setattr(gptscan, '_all_results_cache', [list(original_values)])

    mock_tree.exists.return_value = True
    monkeypatch.setattr(gptscan, '_matches_filter', lambda v: True)
    monkeypatch.setattr(gptscan, '_prepare_tree_row', lambda v: (["wrapped"], ("tag",)))

    new_values = ("path1.py", "90%", "Admin", "User", "95%", "New Snippet")

    gptscan.update_tree_row("item1", new_values)

    assert gptscan._all_results_cache[0] == new_values
    mock_tree.item.assert_called_once_with("item1", values=["wrapped"], tags=("tag",))

def test_update_tree_row_exists_no_longer_matches(mock_tree, monkeypatch):
    original_values = ("path1.py", "80%", "", "", "", "snippet1")
    monkeypatch.setattr(gptscan, '_all_results_cache', [list(original_values)])

    mock_tree.exists.return_value = True
    monkeypatch.setattr(gptscan, '_matches_filter', lambda v: False)

    new_values = ("path1.py", "10%", "", "", "", "Safe Snippet")

    gptscan.update_tree_row("item1", new_values)

    assert gptscan._all_results_cache[0] == new_values
    mock_tree.delete.assert_called_once_with("item1")

def test_update_tree_row_not_exists_now_matches(mock_tree, monkeypatch):
    original_values = ("path1.py", "10%", "", "", "", "snippet1")
    monkeypatch.setattr(gptscan, '_all_results_cache', [list(original_values)])

    mock_tree.exists.return_value = False
    monkeypatch.setattr(gptscan, '_matches_filter', lambda v: True)

    mock_apply_filter = MagicMock()
    monkeypatch.setattr(gptscan, '_apply_filter', mock_apply_filter)

    new_values = ("path1.py", "80%", "Admin", "User", "85%", "Threat Snippet")

    gptscan.update_tree_row("item1", new_values)

    assert gptscan._all_results_cache[0] == new_values
    mock_apply_filter.assert_called_once()

def test_update_tree_row_not_exists_still_no_match(mock_tree, monkeypatch):
    original_values = ("path1.py", "10%", "", "", "", "snippet1")
    monkeypatch.setattr(gptscan, '_all_results_cache', [list(original_values)])

    mock_tree.exists.return_value = False
    monkeypatch.setattr(gptscan, '_matches_filter', lambda v: False)

    mock_apply_filter = MagicMock()
    monkeypatch.setattr(gptscan, '_apply_filter', mock_apply_filter)

    new_values = ("path1.py", "15%", "", "", "", "Still Safe")

    gptscan.update_tree_row("item1", new_values)

    assert gptscan._all_results_cache[0] == new_values
    mock_tree.item.assert_not_called()
    mock_tree.delete.assert_not_called()
    mock_apply_filter.assert_not_called()

def test_update_tree_row_correct_cache_match():
    path = "test.py"
    entry1 = (path, "50%", "Admin1", "User1", "50%", "Snippet1", "10")
    entry2 = (path, "80%", "Admin2", "User2", "80%", "Snippet2", "20")

    gptscan._all_results_cache = [entry1, entry2]
    new_entry2 = (path, "90%", "NewAdmin2", "NewUser2", "95%", "Snippet2", "20")

    with patch("gptscan.tree") as mock_tree:
        mock_tree.exists.return_value = True
        gptscan.update_tree_row("item2", new_entry2)

    assert gptscan._all_results_cache[0] == entry1
    assert gptscan._all_results_cache[1] == new_entry2

def test_update_tree_row_handles_missing_line_indices():
    path = "legacy.py"
    entry1 = (path, "50%", "A1", "U1", "50%", "S1")

    gptscan._all_results_cache = [entry1]
    new_entry1 = (path, "60%", "NewA1", "NewU1", "65%", "S1")

    with patch("gptscan.tree") as mock_tree:
        mock_tree.exists.return_value = True
        gptscan.update_tree_row("item1", new_entry1)

    assert gptscan._all_results_cache[0] == new_entry1
