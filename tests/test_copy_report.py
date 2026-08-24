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

def test_copy_as_csv_details_logic(mock_view_details_env):
    """Test that copy_as_csv_details inside view_details correctly formats single item data as CSV and updates status bar."""
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "I002", "detail.py", own_conf="80%", admin="Admin note", user="User note", gpt_conf="75%", snippet="os.system('rm -rf /')", line=5)

    from gptscan import root as mock_root

    assert "menu_Copy as CSV" in captured
    copy_csv_cmd = captured["menu_Copy as CSV"]

    # Trigger command directly from view_details menu
    copy_csv_cmd()

    mock_root.clipboard_clear.assert_called()
    assert mock_root.clipboard_append.called
    copied_csv = mock_root.clipboard_append.call_args[0][0]
    assert "path,line,own_conf,gpt_conf,admin_desc,end-user_desc,snippet" in copied_csv
    assert "detail.py,5,80%,75%,Admin note,User note,os.system('rm -rf /')" in copied_csv

    status_bar = captured['labels'][0]
    assert status_bar.config_data.get('text') == "Result copied as CSV."


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


def test_copy_as_csv_logic(monkeypatch):
    """Test that copy_as_csv correctly formats selected data as CSV and appends to clipboard."""
    mock_tree = MagicMock()
    mock_tree.selection.return_value = ["I001"]
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    test_results = [{
        "path": "test.py",
        "line": "10",
        "own_conf": "90%",
        "gpt_conf": "85%",
        "admin_desc": "Dangerous code found",
        "end-user_desc": "Highly suspicious",
        "snippet": "eval(input())"
    }]
    monkeypatch.setattr(gptscan, '_get_tree_results_as_dicts', lambda items: test_results)

    mock_update_status = MagicMock()
    monkeypatch.setattr(gptscan, 'update_status', mock_update_status)

    gptscan.copy_as_csv()

    mock_tree.clipboard_clear.assert_called_once()
    assert mock_tree.clipboard_append.call_count == 1
    copied_content = mock_tree.clipboard_append.call_args[0][0]
    assert "path,line,own_conf,gpt_conf,admin_desc,end-user_desc,snippet" in copied_content
    assert "test.py,10,90%,85%,Dangerous code found,Highly suspicious,eval(input())" in copied_content
    mock_update_status.assert_called_once_with("Copied 1 item(s) as CSV.")


def test_copy_as_yaml_logic(monkeypatch):
    """Test that copy_as_yaml correctly formats selected data as YAML and appends to clipboard."""
    mock_tree = MagicMock()
    mock_tree.selection.return_value = ["I001"]
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    test_results = [{
        "path": "test.py",
        "line": "10",
        "own_conf": "90%",
        "gpt_conf": "85%",
        "admin_desc": "Dangerous code found",
        "end-user_desc": "Highly suspicious",
        "snippet": "eval(input())"
    }]
    monkeypatch.setattr(gptscan, '_get_tree_results_as_dicts', lambda items: test_results)
    monkeypatch.setattr(gptscan, 'generate_yaml', lambda results: "yaml_content_mock")

    mock_update_status = MagicMock()
    monkeypatch.setattr(gptscan, 'update_status', mock_update_status)

    gptscan.copy_as_yaml()

    mock_tree.clipboard_clear.assert_called_once()
    mock_tree.clipboard_append.assert_called_once_with("yaml_content_mock")
    mock_update_status.assert_called_once_with("Copied 1 item(s) as YAML.")


def test_copy_as_xml_logic(monkeypatch):
    """Test that copy_as_xml correctly formats selected data as XML and appends to clipboard."""
    mock_tree = MagicMock()
    mock_tree.selection.return_value = ["I001"]
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    test_results = [{
        "path": "test.py",
        "line": "10",
        "own_conf": "90%",
        "gpt_conf": "85%",
        "admin_desc": "Dangerous code found",
        "end-user_desc": "Highly suspicious",
        "snippet": "eval(input())"
    }]
    monkeypatch.setattr(gptscan, '_get_tree_results_as_dicts', lambda items: test_results)
    monkeypatch.setattr(gptscan, 'generate_xml', lambda results: "<xml>content_mock</xml>")

    mock_update_status = MagicMock()
    monkeypatch.setattr(gptscan, 'update_status', mock_update_status)

    gptscan.copy_as_xml()

    mock_tree.clipboard_clear.assert_called_once()
    mock_tree.clipboard_append.assert_called_once_with("<xml>content_mock</xml>")
    mock_update_status.assert_called_once_with("Copied 1 item(s) as XML.")


def test_copy_as_html_logic(monkeypatch):
    """Test that copy_as_html correctly formats selected data as HTML and appends to clipboard."""
    mock_tree = MagicMock()
    mock_tree.selection.return_value = ["I001"]
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    test_results = [{
        "path": "test.py",
        "line": "10",
        "own_conf": "90%",
        "gpt_conf": "85%",
        "admin_desc": "Dangerous code found",
        "end-user_desc": "Highly suspicious",
        "snippet": "eval(input())"
    }]
    monkeypatch.setattr(gptscan, '_get_tree_results_as_dicts', lambda items: test_results)
    monkeypatch.setattr(gptscan, 'generate_html', lambda results: "<html>content_mock</html>")

    mock_update_status = MagicMock()
    monkeypatch.setattr(gptscan, 'update_status', mock_update_status)

    gptscan.copy_as_html()

    mock_tree.clipboard_clear.assert_called_once()
    mock_tree.clipboard_append.assert_called_once_with("<html>content_mock</html>")
    mock_update_status.assert_called_once_with("Copied 1 item(s) as HTML.")


def test_copy_as_html_details_logic(mock_view_details_env):
    """Test that copy_as_html_details inside view_details correctly formats single item data as HTML and updates status bar."""
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "I002", "detail.py", own_conf="80%", admin="Admin note", user="User note", gpt_conf="75%", snippet="os.system('rm -rf /')", line=5)

    from gptscan import root as mock_root

    assert "menu_Copy as HTML" in captured
    copy_html_cmd = captured["menu_Copy as HTML"]

    copy_html_cmd()

    mock_root.clipboard_clear.assert_called()
    assert mock_root.clipboard_append.called
    copied_html = mock_root.clipboard_append.call_args[0][0]
    assert "detail.py" in copied_html

    status_bar = captured['labels'][0]
    assert status_bar.config_data.get('text') == "Result copied as HTML."
