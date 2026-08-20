from unittest.mock import MagicMock
import gptscan

def test_on_target_selected_shifts_focus(monkeypatch):
    """Verify that on_target_selected shifts focus to scan_button."""
    mock_scan_button = MagicMock()
    monkeypatch.setattr(gptscan, "scan_button", mock_scan_button)

    gptscan.on_target_selected()

    mock_scan_button.focus_set.assert_called_once()

def test_on_target_selected_handles_none_scan_button(monkeypatch):
    """Verify that on_target_selected handles scan_button being None gracefully."""
    monkeypatch.setattr(gptscan, "scan_button", None)

    # Should not raise any exception
    gptscan.on_target_selected()
