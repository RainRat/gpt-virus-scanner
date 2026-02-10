import pytest
from unittest.mock import MagicMock, patch
import gptscan

def test_clear_results(monkeypatch):
    # Setup mocks
    mock_tree = MagicMock()
    mock_tree.get_children.return_value = ["item1", "item2"]
    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)

    mock_status_label = MagicMock()
    monkeypatch.setattr(gptscan, 'status_label', mock_status_label, raising=False)

    mock_progress_bar = MagicMock()
    # Mock __setitem__ for progress_bar["value"] = 0
    mock_progress_bar.__setitem__ = MagicMock()
    monkeypatch.setattr(gptscan, 'progress_bar', mock_progress_bar, raising=False)

    # Mock update_tree_columns
    mock_update_cols = MagicMock()
    monkeypatch.setattr(gptscan, 'update_tree_columns', mock_update_cols)

    # Call function
    gptscan.clear_results()

    # Assert
    mock_tree.delete.assert_called_with("item1", "item2")
    mock_status_label.config.assert_called_with(text="Ready")
    mock_progress_bar.__setitem__.assert_called_with("value", 0)
    mock_update_cols.assert_called_once()

def test_show_about(monkeypatch):
    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_msgbox)

    # Action
    gptscan.show_about()

    # Assert
    mock_msgbox.showinfo.assert_called_once()
    args, kwargs = mock_msgbox.showinfo.call_args
    assert "GPT Virus Scanner" in args[1]
    assert gptscan.Config.VERSION in args[1]
