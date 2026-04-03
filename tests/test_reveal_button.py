import pytest
from unittest.mock import MagicMock
import gptscan

def test_reveal_button_management(monkeypatch):
    """Test that reveal_button is managed by set_scanning_state and update_button_states."""
    mock_reveal_button = MagicMock()
    mock_tree = MagicMock()

    monkeypatch.setattr(gptscan, 'reveal_button', mock_reveal_button)
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    # Mock other buttons to avoid None errors if they are used without check
    monkeypatch.setattr(gptscan, 'view_button', MagicMock())
    monkeypatch.setattr(gptscan, 'rescan_button', MagicMock())
    monkeypatch.setattr(gptscan, 'analyze_button', MagicMock())
    monkeypatch.setattr(gptscan, 'exclude_button', MagicMock())
    monkeypatch.setattr(gptscan, 'results_button', MagicMock())
    monkeypatch.setattr(gptscan, 'scan_button', MagicMock())
    monkeypatch.setattr(gptscan, 'cancel_button', MagicMock())

    # Test set_scanning_state(True)
    gptscan.set_scanning_state(True)
    mock_reveal_button.config.assert_called_with(state="disabled")

    # Test set_scanning_state(False)
    gptscan.set_scanning_state(False)
    mock_reveal_button.config.assert_called_with(state="normal")

    # Test update_button_states with selection
    mock_tree.selection.return_value = ("item1",)
    gptscan.update_button_states()
    mock_reveal_button.config.assert_called_with(state="normal")

    # Test update_button_states without selection
    mock_tree.selection.return_value = ()
    gptscan.update_button_states()
    mock_reveal_button.config.assert_called_with(state="disabled")
