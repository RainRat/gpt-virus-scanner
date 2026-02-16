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
    raw_data1 = [path1, "90%", "Malicious", "Dangerous", "95%", "print('evil')"]

    def mock_item(iid, option=None):
        data = {
            "item1": {"values": ["bad1.py", "90%", "Malicious", "Dangerous", "95%", "print('evil')", json.dumps(raw_data1)]},
            "item2": {"values": ["bad2.py", "80%", "Sus", "Bad", "", "exec('code')", ""]} # No hidden column
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
        (path1, "90%", "Malicious", "Dangerous", "95%", "print('evil')"),
        (path2, "80%", "Sus", "Bad", "", "exec('code')"),
        ("safe.py", "5%", "", "", "", "print('hi')")
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
