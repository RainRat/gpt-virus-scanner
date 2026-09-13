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

def test_copy_as_tsv_details_logic(mock_view_details_env):
    """Test that copy_as_tsv_details inside view_details correctly formats single item data as TSV and updates status bar."""
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "I002", "detail.py", own_conf="80%", admin="Admin note", user="User note", gpt_conf="75%", snippet="os.system('rm -rf /')", line=5)

    from gptscan import root as mock_root

    assert "menu_Copy as TSV" in captured
    copy_tsv_cmd = captured["menu_Copy as TSV"]

    # Trigger command directly from view_details menu
    copy_tsv_cmd()

    mock_root.clipboard_clear.assert_called()
    assert mock_root.clipboard_append.called
    copied_tsv = mock_root.clipboard_append.call_args[0][0]
    assert "path\tline\town_conf\tgpt_conf\tadmin_desc\tend-user_desc\tsnippet" in copied_tsv
    assert "detail.py\t5\t80%\t75%\tAdmin note\tUser note\tos.system('rm -rf /')" in copied_tsv

    status_bar = captured['labels'][0]
    assert status_bar.config_data.get('text') == "Result copied as TSV."


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


def test_copy_as_sarif_logic(monkeypatch):
    """Test that copy_as_sarif correctly formats selected data as SARIF and appends to clipboard."""
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
    monkeypatch.setattr(gptscan, 'generate_sarif', lambda results: {"version": "2.1.0", "runs": []})

    mock_update_status = MagicMock()
    monkeypatch.setattr(gptscan, 'update_status', mock_update_status)

    gptscan.copy_as_sarif()

    mock_tree.clipboard_clear.assert_called_once()
    assert mock_tree.clipboard_append.call_count == 1
    copied_sarif = json.loads(mock_tree.clipboard_append.call_args[0][0])
    assert copied_sarif.get("version") == "2.1.0"
    mock_update_status.assert_called_once_with("Copied 1 item(s) as SARIF.")


def test_copy_as_markdown_details_logic(mock_view_details_env):
    """Test copy_as_markdown_details inside view_details."""
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "I002", "detail.py", own_conf="80%", admin="Admin note", user="User note", gpt_conf="75%", snippet="os.system('rm -rf /')", line=5)

    from gptscan import root as mock_root

    assert "menu_Copy as Markdown" in captured
    captured["menu_Copy as Markdown"]()

    mock_root.clipboard_clear.assert_called()
    assert mock_root.clipboard_append.called
    copied_md = mock_root.clipboard_append.call_args[0][0]
    assert "detail.py" in copied_md

    status_bar = captured['labels'][0]
    assert status_bar.config_data.get('text') == "Result copied as Markdown."


def test_copy_as_yaml_details_logic(mock_view_details_env):
    """Test copy_as_yaml_details inside view_details."""
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "I002", "detail.py", own_conf="80%", admin="Admin note", user="User note", gpt_conf="75%", snippet="os.system('rm -rf /')", line=5)

    from gptscan import root as mock_root

    assert "menu_Copy as YAML" in captured
    captured["menu_Copy as YAML"]()

    mock_root.clipboard_clear.assert_called()
    assert mock_root.clipboard_append.called
    copied_yaml = mock_root.clipboard_append.call_args[0][0]
    assert "detail.py" in copied_yaml

    status_bar = captured['labels'][0]
    assert status_bar.config_data.get('text') == "Result copied as YAML."


def test_copy_as_xml_details_logic(mock_view_details_env):
    """Test copy_as_xml_details inside view_details."""
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "I002", "detail.py", own_conf="80%", admin="Admin note", user="User note", gpt_conf="75%", snippet="os.system('rm -rf /')", line=5)

    from gptscan import root as mock_root

    assert "menu_Copy as XML" in captured
    captured["menu_Copy as XML"]()

    mock_root.clipboard_clear.assert_called()
    assert mock_root.clipboard_append.called
    copied_xml = mock_root.clipboard_append.call_args[0][0]
    assert "detail.py" in copied_xml

    status_bar = captured['labels'][0]
    assert status_bar.config_data.get('text') == "Result copied as XML."


def test_copy_as_sarif_details_logic(mock_view_details_env):
    """Test copy_as_sarif_details inside view_details."""
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    setup_details(mock_view_details_env, "I002", "detail.py", own_conf="80%", admin="Admin note", user="User note", gpt_conf="75%", snippet="os.system('rm -rf /')", line=5)

    from gptscan import root as mock_root

    assert "menu_Copy as SARIF" in captured
    captured["menu_Copy as SARIF"]()

    mock_root.clipboard_clear.assert_called()
    assert mock_root.clipboard_append.called
    copied_sarif = mock_root.clipboard_append.call_args[0][0]
    assert "detail.py" in copied_sarif or "2.1.0" in copied_sarif

    status_bar = captured['labels'][0]
    assert status_bar.config_data.get('text') == "Result copied as SARIF."


def test_copy_path_no_tree(monkeypatch):
    """Test copy_path returns early when tree is None."""
    monkeypatch.setattr(gptscan, 'tree', None)
    mock_update_status = MagicMock()
    monkeypatch.setattr(gptscan, 'update_status', mock_update_status)
    gptscan.copy_path()
    mock_update_status.assert_not_called()


def test_copy_path_no_selection(monkeypatch):
    """Test copy_path returns early when no items are selected in tree."""
    mock_tree = MagicMock()
    mock_tree.selection.return_value = []
    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    mock_update_status = MagicMock()
    monkeypatch.setattr(gptscan, 'update_status', mock_update_status)

    gptscan.copy_path()

    mock_tree.clipboard_clear.assert_not_called()
    mock_update_status.assert_not_called()


def test_copy_path_success_single_and_multiple(monkeypatch):
    """Test copy_path copies single and multiple file paths to clipboard."""
    mock_tree = MagicMock()
    mock_tree.selection.return_value = ["item1", "item2"]
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    def mock_get_item_raw_values(item_id):
        if item_id == "item1":
            return ["/path/to/file1.py", "10", "90%", "80%", "admin", "user", "snippet1"]
        elif item_id == "item2":
            return ["/path/to/file2.py", "20", "85%", "75%", "admin2", "user2", "snippet2"]
        return None

    monkeypatch.setattr(gptscan, '_get_item_raw_values', mock_get_item_raw_values)
    mock_update_status = MagicMock()
    monkeypatch.setattr(gptscan, 'update_status', mock_update_status)

    gptscan.copy_path()

    mock_tree.clipboard_clear.assert_called_once()
    mock_tree.clipboard_append.assert_called_once_with("/path/to/file1.py\n/path/to/file2.py")
    mock_update_status.assert_called_once_with("Copied 2 path(s) to clipboard.")


def test_copy_sha256_no_tree_or_selection(monkeypatch):
    """Test copy_sha256 returns early when tree is None or selection is empty."""
    monkeypatch.setattr(gptscan, 'tree', None)
    gptscan.copy_sha256()

    mock_tree = MagicMock()
    mock_tree.selection.return_value = []
    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    mock_update_status = MagicMock()
    monkeypatch.setattr(gptscan, 'update_status', mock_update_status)

    gptscan.copy_sha256()

    mock_tree.clipboard_clear.assert_not_called()
    mock_update_status.assert_not_called()


def test_copy_sha256_single_and_multiple(monkeypatch):
    """Test copy_sha256 formatting for single and multiple selected files."""
    mock_tree = MagicMock()
    mock_tree.selection.return_value = ["item1"]
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    monkeypatch.setattr(gptscan, '_get_item_raw_values', lambda item_id: ["file1.py", "1", "90%", "80%", "a", "u", "snip1"])
    monkeypatch.setattr(gptscan, 'get_effective_sha256', lambda path, snip: "1234567890abcdef")
    mock_update_status = MagicMock()
    monkeypatch.setattr(gptscan, 'update_status', mock_update_status)

    gptscan.copy_sha256()

    mock_tree.clipboard_clear.assert_called_once()
    mock_tree.clipboard_append.assert_called_once_with("1234567890abcdef")
    mock_update_status.assert_called_once_with("SHA256 copied: 12345678...")

    # Multi-item test
    mock_tree.reset_mock()
    mock_update_status.reset_mock()
    mock_tree.selection.return_value = ["item1", "item2"]
    monkeypatch.setattr(gptscan, 'get_effective_sha256', lambda path, snip: f"hash_{path}")

    def mock_raw(item_id):
        return [f"{item_id}.py", "1", "90%", "80%", "a", "u", "snip"]

    monkeypatch.setattr(gptscan, '_get_item_raw_values', mock_raw)

    gptscan.copy_sha256()

    mock_tree.clipboard_clear.assert_called_once()
    mock_tree.clipboard_append.assert_called_once_with("hash_item1.py\nhash_item2.py")
    mock_update_status.assert_called_once_with("Copied 2 SHA256 hashes.")


def test_copy_sha256_failure_warning(monkeypatch):
    """Test copy_sha256 shows warning dialog when hash calculation fails."""
    mock_tree = MagicMock()
    mock_tree.selection.return_value = ["item1"]
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    monkeypatch.setattr(gptscan, '_get_item_raw_values', lambda item_id: ["file1.py", "1", "90%", "80%", "a", "u", "snip1"])
    monkeypatch.setattr(gptscan, 'get_effective_sha256', lambda path, snip: None)
    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan.messagebox, 'showwarning', mock_msgbox)

    gptscan.copy_sha256()

    mock_tree.clipboard_clear.assert_not_called()
    mock_msgbox.assert_called_once_with("Error", "Could not calculate file hashes.")


def test_copy_snippet_no_tree_or_selection(monkeypatch):
    """Test copy_snippet returns early when tree is None or selection is empty."""
    monkeypatch.setattr(gptscan, 'tree', None)
    gptscan.copy_snippet()

    mock_tree = MagicMock()
    mock_tree.selection.return_value = []
    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    mock_update_status = MagicMock()
    monkeypatch.setattr(gptscan, 'update_status', mock_update_status)

    gptscan.copy_snippet()

    mock_tree.clipboard_clear.assert_not_called()
    mock_update_status.assert_not_called()


def test_copy_snippet_single_and_multiple(monkeypatch):
    """Test copy_snippet formatting for single and multiple selections."""
    mock_tree = MagicMock()
    mock_tree.selection.return_value = ["item1"]
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    monkeypatch.setattr(gptscan, '_get_item_raw_values', lambda item_id: ["file1.py", "90%", "80%", "a", "u", "print('hello')", "1"])
    mock_update_status = MagicMock()
    monkeypatch.setattr(gptscan, 'update_status', mock_update_status)

    gptscan.copy_snippet()

    mock_tree.clipboard_clear.assert_called_once()
    mock_tree.clipboard_append.assert_called_once_with("print('hello')")
    mock_update_status.assert_called_once_with("Copied 1 snippets.")

    # Multiple items selection
    mock_tree.reset_mock()
    mock_update_status.reset_mock()
    mock_tree.selection.return_value = ["item1", "item2"]

    def mock_raw(item_id):
        if item_id == "item1":
            return ["file1.py", "90%", "80%", "a", "u", "snip1", "1"]
        return ["file2.py", "85%", "75%", "a", "u", "snip2", "2"]

    monkeypatch.setattr(gptscan, '_get_item_raw_values', mock_raw)

    gptscan.copy_snippet()

    mock_tree.clipboard_clear.assert_called_once()
    mock_tree.clipboard_append.assert_called_once_with("--- file1.py ---\nsnip1\n\n--- file2.py ---\nsnip2")
    mock_update_status.assert_called_once_with("Copied 2 snippets.")
