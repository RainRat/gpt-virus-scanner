from unittest.mock import MagicMock
import gptscan

def test_clear_results(monkeypatch):
    # Mock global variables
    mock_tree = MagicMock()
    mock_progress_bar = MagicMock()
    mock_status_label = MagicMock()
    mock_root = MagicMock()

    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    monkeypatch.setattr(gptscan, 'progress_bar', mock_progress_bar)
    monkeypatch.setattr(gptscan, 'status_label', mock_status_label)
    monkeypatch.setattr(gptscan, 'root', mock_root)

    # Pre-checks
    mock_tree.get_children.return_value = ('item1', 'item2')

    # Call clear_results
    gptscan.clear_results()

    # Assertions
    mock_tree.delete.assert_called_with('item1', 'item2')
    # For progress_bar['value'] = 0
    mock_progress_bar.__setitem__.assert_called_with('value', 0)
    # update_status calls status_label.config and root.update_idletasks
    mock_status_label.config.assert_called_with(text="Ready")
    mock_root.update_idletasks.assert_called()
