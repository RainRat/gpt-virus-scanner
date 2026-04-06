import pytest
from unittest.mock import MagicMock, patch
import gptscan

@pytest.fixture
def mock_tree(monkeypatch):
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    return mock_tree

def test_update_tree_row_exists_matches(mock_tree, monkeypatch):
    """Test updating an existing row that still matches the filter."""
    # Setup cache
    original_values = ("path1.py", "80%", "", "", "", "snippet1")
    monkeypatch.setattr(gptscan, '_all_results_cache', [list(original_values)])

    # Mock behavior
    mock_tree.exists.return_value = True
    monkeypatch.setattr(gptscan, '_matches_filter', lambda v: True)
    monkeypatch.setattr(gptscan, '_prepare_tree_row', lambda v: (["wrapped"], ("tag",)))

    new_values = ("path1.py", "90%", "Admin", "User", "95%", "New Snippet")

    gptscan.update_tree_row("item1", new_values)

    # Verify cache update
    assert gptscan._all_results_cache[0] == new_values

    # Verify tree update
    mock_tree.item.assert_called_once_with("item1", values=["wrapped"], tags=("tag",))

def test_update_tree_row_exists_no_longer_matches(mock_tree, monkeypatch):
    """Test that a row is deleted if it no longer matches the filter."""
    original_values = ("path1.py", "80%", "", "", "", "snippet1")
    monkeypatch.setattr(gptscan, '_all_results_cache', [list(original_values)])

    mock_tree.exists.return_value = True
    # No longer matches filter (e.g. threat level dropped below threshold)
    monkeypatch.setattr(gptscan, '_matches_filter', lambda v: False)

    new_values = ("path1.py", "10%", "", "", "", "Safe Snippet")

    gptscan.update_tree_row("item1", new_values)

    # Verify cache update
    assert gptscan._all_results_cache[0] == new_values

    # Verify tree deletion
    mock_tree.delete.assert_called_once_with("item1")

def test_update_tree_row_not_exists_now_matches(mock_tree, monkeypatch):
    """Test that _apply_filter is called when a hidden item now matches."""
    original_values = ("path1.py", "10%", "", "", "", "snippet1")
    monkeypatch.setattr(gptscan, '_all_results_cache', [list(original_values)])

    # Item is not in the tree (hidden by filter)
    mock_tree.exists.return_value = False
    # Now matches filter (e.g. AI analysis increased threat level)
    monkeypatch.setattr(gptscan, '_matches_filter', lambda v: True)

    mock_apply_filter = MagicMock()
    monkeypatch.setattr(gptscan, '_apply_filter', mock_apply_filter)

    new_values = ("path1.py", "80%", "Admin", "User", "85%", "Threat Snippet")

    gptscan.update_tree_row("item1", new_values)

    # Verify cache update
    assert gptscan._all_results_cache[0] == new_values

    # Verify _apply_filter was called to refresh the view
    mock_apply_filter.assert_called_once()

def test_update_tree_row_not_exists_still_no_match(mock_tree, monkeypatch):
    """Test that nothing happens if item doesn't exist and still doesn't match."""
    original_values = ("path1.py", "10%", "", "", "", "snippet1")
    monkeypatch.setattr(gptscan, '_all_results_cache', [list(original_values)])

    mock_tree.exists.return_value = False
    monkeypatch.setattr(gptscan, '_matches_filter', lambda v: False)

    mock_apply_filter = MagicMock()
    monkeypatch.setattr(gptscan, '_apply_filter', mock_apply_filter)

    new_values = ("path1.py", "15%", "", "", "", "Still Safe")

    gptscan.update_tree_row("item1", new_values)

    # Verify cache update
    assert gptscan._all_results_cache[0] == new_values

    # Verify no UI actions
    mock_tree.item.assert_not_called()
    mock_tree.delete.assert_not_called()
    mock_apply_filter.assert_not_called()
