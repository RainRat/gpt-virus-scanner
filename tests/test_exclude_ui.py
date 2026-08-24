import os
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import gptscan

@pytest.fixture
def mock_gui(monkeypatch):
    """Mock all GUI-related globals and methods."""
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, "tree", mock_tree)

    mock_askyesno = MagicMock(return_value=True)
    monkeypatch.setattr("tkinter.messagebox.askyesno", mock_askyesno)

    mock_apply_filter = MagicMock()
    monkeypatch.setattr("gptscan._apply_filter", mock_apply_filter)

    mock_update_status = MagicMock()
    monkeypatch.setattr("gptscan.update_status", mock_update_status)

    return {
        "tree": mock_tree,
        "askyesno": mock_askyesno,
        "apply_filter": mock_apply_filter,
        "update_status": mock_update_status
    }

def test_exclude_selected_logic(tmp_path, monkeypatch, mock_gui):
    """Test the core logic of exclude_selected without a full GUI."""
    # Setup
    monkeypatch.chdir(tmp_path)
    # Ensure gptscan uses the tmp_path for .gptscanignore
    ignore_file = tmp_path / ".gptscanignore"

    # Mock selection: two items
    mock_gui["tree"].selection.return_value = ["item1", "item2"]

    # Mock item values
    # Item 1 has raw data in hidden column
    path1 = str(tmp_path / "bad1.py")
    raw_data1 = [path1, "90%", "Malicious", "Dangerous", "95%", "print('evil')", "1"]

    def mock_item(iid, option=None):
        data = {
            "item1": {"values": ["bad1.py", "90%", "Malicious", "Dangerous", "95%", "print('evil')", "1", json.dumps(raw_data1)]},
            "item2": {"values": ["bad2.py", "80%", "Sus", "Bad", "", "exec('code')", "1", ""]} # No hidden column
        }
        if option == "values":
            return data[iid]["values"]
        return data[iid]

    mock_gui["tree"].item.side_effect = mock_item

    # Mock Config
    monkeypatch.setattr(gptscan.Config, "ignore_patterns", [])

    # Mock _all_results_cache
    path2 = "bad2.py"
    gptscan._all_results_cache = [
        (path1, "90%", "Malicious", "Dangerous", "95%", "print('evil')", "1"),
        (path2, "80%", "Sus", "Bad", "", "exec('code')", "1"),
        ("safe.py", "5%", "", "", "", "print('hi')", "1")
    ]

    # Execute
    gptscan.exclude_selected()

    # Verify .gptscanignore
    assert ignore_file.exists()
    content = ignore_file.read_text()
    assert "bad2.py" in content

    # Verify Config.ignore_patterns
    assert len(gptscan.Config.ignore_patterns) == 2

    # Verify cache update
    assert len(gptscan._all_results_cache) == 1
    assert gptscan._all_results_cache[0][0] == "safe.py"

    # Verify UI calls
    mock_gui["apply_filter"].assert_called_once()
    mock_gui["update_status"].assert_called_with("Excluded 2 file(s).")

def test_exclude_selected_cancelled(monkeypatch, mock_gui):
    """Test that nothing happens if user cancels the confirmation."""
    mock_gui["tree"].selection.return_value = ["item1"]
    mock_gui["askyesno"].return_value = False

    # Track cache
    initial_cache = [("a.py", "1%", "", "", "", "")]
    gptscan._all_results_cache = list(initial_cache)

    gptscan.exclude_selected()

    assert gptscan._all_results_cache == initial_cache

def test_exclude_selected_no_selection(monkeypatch, mock_gui):
    """Test that nothing happens if nothing is selected."""
    mock_gui["tree"].selection.return_value = []

    gptscan.exclude_selected()

    mock_gui["askyesno"].assert_not_called()


def test_exclude_selected_advances_selection(mock_gui, monkeypatch):
    """Verify that exclude_selected advances selection to the next item."""
    # Mock exclude_paths to return True (success)
    monkeypatch.setattr("gptscan.exclude_paths", MagicMock(return_value=True))

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
    mock_gui["tree"].focus_set.assert_called_once()


def test_exclude_selected_handles_last_item(mock_gui, monkeypatch):
    """Verify that exclude_selected selects the new last item if the end is reached."""
    # Mock exclude_paths to return True (success)
    monkeypatch.setattr("gptscan.exclude_paths", MagicMock(return_value=True))

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


def test_exclude_selected_handles_empty_result(mock_gui, monkeypatch):
    """Verify that exclude_selected handles the case where no items remain."""
    # Mock exclude_paths to return True (success)
    monkeypatch.setattr("gptscan.exclude_paths", MagicMock(return_value=True))

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


def test_exclude_paths_consolidated_logic(tmp_path, monkeypatch):
    # Set up a dummy .gptscanignore
    ignore_file = tmp_path / ".gptscanignore"
    ignore_file.write_text("existing.py\n")

    # Change CWD to tmp_path so relpath works predictably
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "exists", lambda p: str(p) == ".gptscanignore" or p == Path(".gptscanignore"))

    # Mock Config
    monkeypatch.setattr(gptscan.Config, "ignore_patterns", ["existing.py"])

    # Mock messagebox
    monkeypatch.setattr(gptscan.messagebox, "askyesno", lambda title, msg: True)

    # Mock _apply_filter and update_status
    monkeypatch.setattr(gptscan, "_apply_filter", MagicMock())
    monkeypatch.setattr(gptscan, "update_status", MagicMock())

    # Initialize cache
    gptscan._all_results_cache = [
        ("file1.py", "10%", "", "", "", "print(1)", "1"),
        ("file2.py", "20%", "", "", "", "print(2)", "1"),
        ("existing.py", "30%", "", "", "", "print(3)", "1")
    ]

    # Test exclusion
    paths_to_exclude = ["file1.py", "file2.py"]

    # We need to mock open specifically for .gptscanignore in the current directory
    # But it's easier to just let it use the real filesystem in tmp_path since we chdir'd

    result = gptscan.exclude_paths(paths_to_exclude, confirm=False)

    assert result is True

    # Check .gptscanignore content
    content = ignore_file.read_text()
    assert "file1.py" in content
    assert "file2.py" in content
    assert content.count("file1.py") == 1

    # Check Config.ignore_patterns
    assert "file1.py" in gptscan.Config.ignore_patterns
    assert "file2.py" in gptscan.Config.ignore_patterns

    # Check _all_results_cache
    assert len(gptscan._all_results_cache) == 1
    assert gptscan._all_results_cache[0][0] == "existing.py"
