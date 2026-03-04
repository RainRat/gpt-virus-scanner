import json
from unittest.mock import MagicMock
import pytest
import gptscan

@pytest.fixture
def mock_gui(monkeypatch):
    """Mock GUI globals and relevant methods."""
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, "tree", mock_tree)

    # Mock exclude_paths to return True (success)
    monkeypatch.setattr("gptscan.exclude_paths", MagicMock(return_value=True))

    return {"tree": mock_tree}

def test_exclude_selected_advances_selection(mock_gui):
    """Verify that exclude_selected advances selection to the next item."""
    # initial items: item1, item2, item3
    # user selects item2 and excludes it
    # expected: item3 is selected

    initial_items = ["item1", "item2", "item3"]
    mock_gui["tree"].get_children.side_effect = [
        initial_items,            # Before exclusion
        ["item1", "item3"]        # After exclusion
    ]
    mock_gui["tree"].selection.return_value = ["item2"]

    # Mock item values to satisfy exclude_selected's data gathering
    def mock_item(iid, option=None):
        return {"values": [f"{iid}.py", "10%", "", "", "", "print('hi')", "1", ""]}
    mock_gui["tree"].item.side_effect = mock_item

    gptscan.exclude_selected()

    # Verify new selection is item3 (which was at index 2, now at index 1)
    mock_gui["tree"].selection_set.assert_called_with("item3")
    mock_gui["tree"].focus.assert_called_with("item3")
    mock_gui["tree"].see.assert_called_with("item3")

def test_exclude_selected_handles_last_item(mock_gui):
    """Verify that exclude_selected selects the new last item if the end is reached."""
    # initial items: item1, item2
    # user selects item2 and excludes it
    # expected: item1 is selected

    initial_items = ["item1", "item2"]
    mock_gui["tree"].get_children.side_effect = [
        initial_items,     # Before exclusion
        ["item1"]          # After exclusion
    ]
    mock_gui["tree"].selection.return_value = ["item2"]

    # Mock item values
    mock_gui["tree"].item.return_value = {"values": ["item2.py", "10%", "", "", "", "", "1", ""]}

    gptscan.exclude_selected()

    # Index of item2 was 1. New list has length 1. min(1, 1-1) = 0.
    # item1 (index 0) should be selected.
    mock_gui["tree"].selection_set.assert_called_with("item1")

def test_exclude_selected_handles_empty_result(mock_gui):
    """Verify that exclude_selected handles the case where no items remain."""
    initial_items = ["item1"]
    mock_gui["tree"].get_children.side_effect = [
        initial_items,     # Before
        []                 # After
    ]
    mock_gui["tree"].selection.return_value = ["item1"]
    mock_gui["tree"].item.return_value = {"values": ["item1.py", "10%", "", "", "", "", "1", ""]}

    # Should not crash
    gptscan.exclude_selected()

    # selection_set should not be called if list is empty
    mock_gui["tree"].selection_set.assert_not_called()
