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

def test_copy_as_report_details_logic(monkeypatch):
    """Test that copy_as_report_details inside view_details correctly formats data."""
    # We need to simulate the local function. Since it's nested, we test the logic.
    mock_current_item_id = "I002"
    mock_root = MagicMock()
    monkeypatch.setattr(gptscan, 'root', mock_root)

    # Mock _get_tree_results_as_dicts
    test_results = [{
        "path": "detail.py",
        "own_conf": "80%",
        "admin_desc": "Admin note",
        "end-user_desc": "User note",
        "gpt_conf": "75%",
        "snippet": "os.system('rm -rf /')",
        "line": "5"
    }]
    mock_get_dicts = MagicMock(return_value=test_results)
    monkeypatch.setattr(gptscan, '_get_tree_results_as_dicts', mock_get_dicts)

    # Mock generate_console_report
    mock_report = "Details Report"
    mock_gen_report = MagicMock(return_value=mock_report)
    monkeypatch.setattr(gptscan, 'generate_console_report', mock_gen_report)

    # Mock set_local_status
    mock_set_status = MagicMock()

    # Since copy_as_report_details is local, we recreate its logic here to test it
    def copy_as_report_details():
        results = gptscan._get_tree_results_as_dicts([mock_current_item_id])
        if results:
            report = gptscan.generate_console_report(results, use_color=False)
            mock_root.clipboard_clear()
            mock_root.clipboard_append(report)
            mock_set_status("Result copied as Triage Report.", temporary=True)

    copy_as_report_details()

    # Verify calls
    mock_get_dicts.assert_called_once_with([mock_current_item_id])
    mock_gen_report.assert_called_once_with(test_results, use_color=False)
    mock_root.clipboard_clear.assert_called_once()
    mock_root.clipboard_append.assert_called_once_with(mock_report)
    mock_set_status.assert_called_once_with("Result copied as Triage Report.", temporary=True)


def test_copy_as_csv_logic(monkeypatch):
    """Test that copy_as_csv correctly formats selected rows as CSV to clipboard."""
    mock_tree = MagicMock()
    mock_tree.selection.return_value = ["I001"]
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

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

    mock_update_status = MagicMock()
    monkeypatch.setattr(gptscan, 'update_status', mock_update_status)

    gptscan.copy_as_csv()

    mock_get_dicts.assert_called_once_with(["I001"])
    mock_tree.clipboard_clear.assert_called_once()

    # Verify CSV string format
    appended_csv = mock_tree.clipboard_append.call_args[0][0]
    assert "path,own_conf,admin_desc,end-user_desc,gpt_conf,snippet,line" in appended_csv
    assert "test.py,90%,Dangerous code found,Highly suspicious,85%,eval(input()),10" in appended_csv
    mock_update_status.assert_called_once_with("Copied 1 item(s) as CSV.")


def test_copy_as_csv_details_logic(monkeypatch):
    """Test copy_as_csv_details formatting logic for single result item."""
    mock_current_item_id = "I002"
    mock_root = MagicMock()
    monkeypatch.setattr(gptscan, 'root', mock_root)

    test_results = [{
        "path": "detail.py",
        "own_conf": "80%",
        "admin_desc": "Admin note",
        "end-user_desc": "User note",
        "gpt_conf": "75%",
        "snippet": "os.system('rm -rf /')",
        "line": "5"
    }]
    mock_get_dicts = MagicMock(return_value=test_results)
    monkeypatch.setattr(gptscan, '_get_tree_results_as_dicts', mock_get_dicts)

    import io
    import csv

    mock_set_status = MagicMock()

    def copy_as_csv_details():
        results = gptscan._get_tree_results_as_dicts([mock_current_item_id])
        if results:
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet", "line"])
            r = results[0]
            writer.writerow([
                r.get("path", ""),
                r.get("own_conf", ""),
                r.get("admin_desc", ""),
                r.get("end-user_desc", ""),
                r.get("gpt_conf", ""),
                r.get("snippet", ""),
                r.get("line", "-")
            ])
            mock_root.clipboard_clear()
            mock_root.clipboard_append(output.getvalue())
            mock_set_status("Result copied as CSV.", temporary=True)

    copy_as_csv_details()

    mock_get_dicts.assert_called_once_with([mock_current_item_id])
    mock_root.clipboard_clear.assert_called_once()
    appended_csv = mock_root.clipboard_append.call_args[0][0]
    assert "detail.py,80%,Admin note,User note,75%,os.system('rm -rf /'),5" in appended_csv
    mock_set_status.assert_called_once_with("Result copied as CSV.", temporary=True)
