from unittest.mock import MagicMock
import pytest
import gptscan

def test_finish_scan_state_summary_found(monkeypatch):
    # Setup
    mock_status_label = MagicMock()
    monkeypatch.setattr(gptscan, 'status_label', mock_status_label, raising=False)
    monkeypatch.setattr(gptscan, 'root', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'current_cancel_event', None)

    # Action
    gptscan.finish_scan_state(found_count=5, total_files=10)

    # Assert
    mock_status_label.config.assert_called_with(text="Scan complete. Found 5 suspicious files out of 10 scanned.")

def test_finish_scan_state_summary_one_found(monkeypatch):
    # Setup
    mock_status_label = MagicMock()
    monkeypatch.setattr(gptscan, 'status_label', mock_status_label, raising=False)
    monkeypatch.setattr(gptscan, 'root', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'current_cancel_event', None)

    # Action
    gptscan.finish_scan_state(found_count=1, total_files=10)

    # Assert
    mock_status_label.config.assert_called_with(text="Scan complete. Found 1 suspicious file out of 10 scanned.")

def test_finish_scan_state_summary_none_found(monkeypatch):
    # Setup
    mock_status_label = MagicMock()
    monkeypatch.setattr(gptscan, 'status_label', mock_status_label, raising=False)
    monkeypatch.setattr(gptscan, 'root', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'current_cancel_event', None)

    # Action
    gptscan.finish_scan_state(found_count=0, total_files=10)

    # Assert
    mock_status_label.config.assert_called_with(text="Scan complete. No suspicious files found out of 10 scanned.")

def test_finish_scan_state_cancelled(monkeypatch):
    # Setup
    mock_status_label = MagicMock()
    monkeypatch.setattr(gptscan, 'status_label', mock_status_label, raising=False)
    monkeypatch.setattr(gptscan, 'root', MagicMock(), raising=False)

    mock_event = MagicMock()
    mock_event.is_set.return_value = True
    monkeypatch.setattr(gptscan, 'current_cancel_event', mock_event)

    # Action
    gptscan.finish_scan_state(found_count=5, total_files=10)

    # Assert
    mock_status_label.config.assert_called_with(text="Scan cancelled.")

def test_button_click_clears_tree(monkeypatch):
    # Setup
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = "/some/path"
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox, raising=False)
    monkeypatch.setattr(gptscan, 'current_cancel_event', None)
    monkeypatch.setattr(gptscan, 'status_label', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'root', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan.os.path, 'exists', lambda p: True)
    monkeypatch.setattr(gptscan.threading, 'Thread', MagicMock())

    # Mock vars
    for var_name in ['deep_var', 'all_var', 'gpt_var', 'dry_var']:
        mock_var = MagicMock()
        mock_var.get.return_value = False
        monkeypatch.setattr(gptscan, var_name, mock_var, raising=False)

    # Mock tree with children
    mock_tree = MagicMock()
    mock_tree.get_children.return_value = ["item1", "item2"]
    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)

    # Action
    gptscan.button_click()

    # Assert
    assert mock_tree.delete.call_count == 2
    mock_tree.delete.assert_any_call("item1")
    mock_tree.delete.assert_any_call("item2")
