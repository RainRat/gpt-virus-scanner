import pytest
from unittest.mock import MagicMock, patch
import gptscan
import json

def test_copy_as_report_logic(monkeypatch):
    """Test that copy_as_report correctly formats selected data and appends to clipboard."""
    mock_tree = MagicMock()
    mock_tree.selection.return_value = ["I001"]
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    # Mock _get_tree_results_as_dicts
    test_results = [{
        "path": "test.py",
        "own_conf": "90%",
        "admin_desc": "Dangerous code found",
        "end-user_desc": "Highly suspicious",
        "gpt_conf": "85%",
        "snippet": "eval(input())",
        "line": "10"
    }]
    mock_get_dicts = MagicMock(return_value=test_results)
    monkeypatch.setattr(gptscan, '_get_tree_results_as_dicts', mock_get_dicts)

    # Mock generate_console_report
    mock_report = "Mocked Report"
    mock_gen_report = MagicMock(return_value=mock_report)
    monkeypatch.setattr(gptscan, 'generate_console_report', mock_gen_report)

    # Mock clipboard and status update
    mock_update_status = MagicMock()
    monkeypatch.setattr(gptscan, 'update_status', mock_update_status)

    # Call copy_as_report
    gptscan.copy_as_report()

    # Verify calls
    mock_get_dicts.assert_called_once_with(["I001"])
    mock_gen_report.assert_called_once_with(test_results, use_color=False)
    mock_tree.clipboard_clear.assert_called_once()
    mock_tree.clipboard_append.assert_called_once_with(mock_report)
    mock_update_status.assert_called_once_with("Copied 1 item(s) as Triage Report.")

from tests.test_view_details import mock_view_details_env, setup_details

def test_copy_as_report_details_logic(mock_view_details_env):
    """Test that copy_as_report_details inside view_details correctly formats data and updates status bar."""
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "I002", "detail.py", own_conf="80%", admin="Admin note", user="User note", gpt_conf="75%", snippet="os.system('rm -rf /')", line=5)

    from gptscan import root as mock_root

    assert "menu_Copy as Triage Report" in captured
    copy_report_cmd = captured["menu_Copy as Triage Report"]

    # Trigger command directly from view_details menu
    copy_report_cmd()

    mock_root.clipboard_clear.assert_called()
    assert mock_root.clipboard_append.called
    copied_report = mock_root.clipboard_append.call_args[0][0]
    assert "detail.py" in copied_report

    status_bar = captured['labels'][0]
    assert status_bar.config_data.get('text') == "Result copied as Triage Report."
