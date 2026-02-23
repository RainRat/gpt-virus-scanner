import pytest
from unittest.mock import MagicMock, patch
import gptscan

def test_apply_filter_updates_status(monkeypatch):
    """Test that _apply_filter updates the status label with match counts."""
    mock_tree = MagicMock()
    mock_status_label = MagicMock()
    mock_filter_var = MagicMock()

    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    monkeypatch.setattr(gptscan, 'status_label', mock_status_label)
    monkeypatch.setattr(gptscan, 'filter_var', mock_filter_var)
    monkeypatch.setattr(gptscan, 'current_cancel_event', None)
    monkeypatch.setattr(gptscan, '_last_scan_summary', "Original Summary")

    # Setup cache with items
    results = [
        ("danger.py", "90%", "Admin", "User", "80%", "eval(evil)"),
        ("safe.py", "10%", "Admin", "User", "0%", "print(safe)")
    ]
    monkeypatch.setattr(gptscan, '_all_results_cache', results)

    # Mock _prepare_tree_row and _matches_filter to work with our data
    monkeypatch.setattr(gptscan, '_prepare_tree_row', lambda v: (list(v), ()))
    monkeypatch.setattr(gptscan.Config, 'THRESHOLD', 0)

    # 1. Test filtering with results
    mock_filter_var.get.return_value = "danger"
    gptscan._apply_filter()

    # Verify status was updated with "Showing 1 of 2 results matching 'danger'"
    mock_status_label.config.assert_any_call(text="Showing 1 of 2 results matching 'danger'")

    # 2. Test clearing filter restores summary
    mock_filter_var.get.return_value = ""
    gptscan._apply_filter()
    mock_status_label.config.assert_any_call(text="Original Summary")

    # 3. Test filter with no results
    mock_filter_var.get.return_value = "nothing"
    gptscan._apply_filter()
    mock_status_label.config.assert_any_call(text="Showing 0 of 2 results matching 'nothing'")

def test_finish_scan_updates_summary_variable(monkeypatch):
    """Test that finish_scan_state updates the _last_scan_summary global."""
    monkeypatch.setattr(gptscan, 'status_label', MagicMock())
    monkeypatch.setattr(gptscan, 'root', MagicMock())
    monkeypatch.setattr(gptscan, '_last_scan_summary', "")

    # Call finish_scan_state with results
    gptscan.finish_scan_state(total_scanned=10, threats_found=2, total_bytes=1024, elapsed_time=1.0)

    assert "10 files scanned, 2 suspicious files found" in gptscan._last_scan_summary

def test_clear_results_resets_summary_variable(monkeypatch):
    """Test that clear_results resets the _last_scan_summary global."""
    monkeypatch.setattr(gptscan, 'status_label', MagicMock())
    monkeypatch.setattr(gptscan, 'progress_bar', MagicMock())
    monkeypatch.setattr(gptscan, '_last_scan_summary', "Some Summary")

    gptscan.clear_results()

    assert gptscan._last_scan_summary == ""
