import pytest
from unittest.mock import MagicMock
import gptscan

def test_open_button_management(monkeypatch):
    """Test that open_button is managed by set_scanning_state and update_button_states."""
    mock_open_button = MagicMock()
    mock_tree = MagicMock()

    monkeypatch.setattr(gptscan, 'open_button', mock_open_button)
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    # Mock other buttons to avoid None errors if they are used without check
    monkeypatch.setattr(gptscan, 'view_button', MagicMock())
    monkeypatch.setattr(gptscan, 'rescan_button', MagicMock())
    monkeypatch.setattr(gptscan, 'analyze_button', MagicMock())
    monkeypatch.setattr(gptscan, 'exclude_button', MagicMock())
    monkeypatch.setattr(gptscan, 'reveal_button', MagicMock())
    monkeypatch.setattr(gptscan, 'import_button', MagicMock())
    monkeypatch.setattr(gptscan, 'export_button', MagicMock())
    monkeypatch.setattr(gptscan, 'clear_button', MagicMock())
    monkeypatch.setattr(gptscan, 'scan_button', MagicMock())
    monkeypatch.setattr(gptscan, 'cancel_button', MagicMock())

    # Test set_scanning_state(True)
    gptscan.set_scanning_state(True)
    mock_open_button.config.assert_called_with(state="disabled")

    # Test set_scanning_state(False)
    gptscan.set_scanning_state(False)
    # Note: set_scanning_state(False) calls update_button_states()
    # which we test below based on selection.

    # Test update_button_states with selection
    mock_tree.selection.return_value = ("item1",)
    gptscan.update_button_states()
    mock_open_button.config.assert_called_with(state="normal")

    # Test update_button_states without selection
    mock_tree.selection.return_value = ()
    gptscan.update_button_states()
    mock_open_button.config.assert_called_with(state="disabled")
