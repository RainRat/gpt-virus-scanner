import pytest
from unittest.mock import MagicMock, patch
import gptscan

def test_update_tree_row_correct_cache_match():
    # Setup multiple entries for the same path with different line numbers
    path = "test.py"
    # values: (path, own_conf, admin_desc, user_desc, gpt_conf, snippet, line)
    entry1 = (path, "50%", "Admin1", "User1", "50%", "Snippet1", "10")
    entry2 = (path, "80%", "Admin2", "User2", "80%", "Snippet2", "20")

    gptscan._all_results_cache = [entry1, entry2]

    # New data to update entry2 (matching line 20)
    new_entry2 = (path, "90%", "NewAdmin2", "NewUser2", "95%", "Snippet2", "20")

    # Mock tree for the GUI part of update_tree_row
    with patch("gptscan.tree") as mock_tree:
        mock_tree.exists.return_value = True
        # item_id is arbitrary for cache test
        gptscan.update_tree_row("item2", new_entry2)

    # Check that entry1 is UNCHANGED and entry2 IS UPDATED
    assert gptscan._all_results_cache[0] == entry1
    assert gptscan._all_results_cache[1] == new_entry2

def test_update_tree_row_handles_missing_line_indices():
    # Setup entries without line numbers (backwards compatibility or partial data)
    path = "legacy.py"
    entry1 = (path, "50%", "A1", "U1", "50%", "S1") # Length 6

    gptscan._all_results_cache = [entry1]

    new_entry1 = (path, "60%", "NewA1", "NewU1", "65%", "S1")

    with patch("gptscan.tree") as mock_tree:
        mock_tree.exists.return_value = True
        gptscan.update_tree_row("item1", new_entry1)

    # Should still match on path only if line index is missing
    assert gptscan._all_results_cache[0] == new_entry1
