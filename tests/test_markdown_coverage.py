import pytest
from unittest.mock import MagicMock, patch
import gptscan
import json
import os

def test_generate_markdown_empty():
    """Test generate_markdown with empty results (covers line 1413)."""
    assert gptscan.generate_markdown([]) == "# GPT Scan Results\n\nNo suspicious files found."

def test_generate_markdown_long_snippet():
    """Test generate_markdown with snippets longer than 100 characters (covers line 1442)."""
    long_snippet = "A" * 150
    results = [{
        "path": "test.py",
        "own_conf": "90%",
        "gpt_conf": "95%",
        "admin_desc": "Admin",
        "end-user_desc": "User",
        "snippet": long_snippet
    }]
    md = gptscan.generate_markdown(results)
    # The snippet in the table is truncated to 97 chars + ...
    expected_snippet = "A" * 97 + "..."
    assert f"`{expected_snippet}`" in md

def test_generate_markdown_escaping():
    """Test generate_markdown with special characters to ensure proper escaping."""
    results = [{
        "path": "test|file.py",
        "own_conf": "90%",
        "gpt_conf": "95%",
        "admin_desc": "Admin|Note",
        "end-user_desc": "User|Note",
        "snippet": "print('|')"
    }]
    md = gptscan.generate_markdown(results)
    assert "test\\|file.py" in md
    assert "Admin\\|Note" in md
    assert "User\\|Note" in md
    assert "`print('\\|')`" in md

def test_run_cli_markdown(monkeypatch, capsys):
    """Test run_cli with markdown output format (covers line 1547)."""
    def mock_scan_files(*args, **kwargs):
        yield ('result', ("test.py", "90%", "Admin", "User", "95%", "print('hi')"))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    # We need to mock sys.stdout because run_cli might be printing to it directly
    # and capsys captures it.

    gptscan.run_cli(["."], deep=False, show_all=True, use_gpt=True, rate_limit=60, output_format='markdown')

    captured = capsys.readouterr()
    assert "# GPT Scan Results" in captured.out
    assert "| test.py | 95% | **Admin:** Admin<br>**User:** User | `print('hi')` |" in captured.out

def test_export_results_markdown(monkeypatch, tmp_path):
    """Test export_results with .md extension (covers line 1725)."""
    mock_tree = MagicMock()
    mock_tree.__getitem__.return_value = ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet", "orig_json")
    mock_tree.get_children.return_value = ["item1"]

    # Case where orig_json is present
    raw_values = ["test.py", "90%", "Admin", "User", "95%", "print('hi')"]
    vals = ("test.py", "90%", "Admin", "User", "95%", "print('hi')", json.dumps(raw_values))
    def mock_item(iid, option=None):
        if option == "values": return vals
        return {"values": vals}
    mock_tree.item.side_effect = mock_item

    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)
    monkeypatch.setattr(gptscan.messagebox, 'showinfo', MagicMock())

    md_path = tmp_path / "results.md"
    monkeypatch.setattr(gptscan.tkinter.filedialog, 'asksaveasfilename', lambda **k: str(md_path))

    gptscan.export_results()

    assert md_path.exists()
    content = md_path.read_text()
    assert "# GPT Scan Results" in content
    assert "test.py" in content

def test_copy_as_markdown_fallback(monkeypatch):
    """Test copy_as_markdown fallback when orig_json is missing (covers lines 1817-1818)."""
    mock_tree = MagicMock()
    mock_tree.__getitem__.return_value = ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet", "orig_json")
    mock_tree.selection.return_value = ["item1"]

    # Case where orig_json is missing or empty (6 columns only or 7th is empty)
    vals = ("test.py", "90%", "Admin", "User", "95%", "print('hi\nwrapped')")
    def mock_item(iid, option=None):
        if option == "values": return vals
        return {"values": vals}
    mock_tree.item.side_effect = mock_item

    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)

    gptscan.copy_as_markdown()

    mock_tree.clipboard_clear.assert_called_once()
    args, _ = mock_tree.clipboard_append.call_args
    md = args[0]
    # In fallback, newlines are replaced by spaces
    assert "print('hi wrapped')" in md

def test_copy_as_markdown_no_tree(monkeypatch):
    """Test copy_as_markdown when tree is None (covers line 1803)."""
    monkeypatch.setattr(gptscan, 'tree', None, raising=False)
    # Should just return
    gptscan.copy_as_markdown()

def test_copy_as_markdown_no_selection(monkeypatch):
    """Test copy_as_markdown when nothing is selected (covers line 1807)."""
    mock_tree = MagicMock()
    mock_tree.selection.return_value = []
    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)
    # Should just return
    gptscan.copy_as_markdown()

def test_get_selected_row_values_no_tree(monkeypatch):
    """Test _get_selected_row_values when tree is None (covers line 1829)."""
    monkeypatch.setattr(gptscan, 'tree', None, raising=False)
    assert gptscan._get_selected_row_values() is None

def test_export_results_sarif(monkeypatch, tmp_path):
    """Test export_results with .sarif extension (covers lines 1720-1722)."""
    mock_tree = MagicMock()
    mock_tree.__getitem__.return_value = ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet", "orig_json")
    mock_tree.get_children.return_value = ["item1"]

    vals = ("test.py", "90%", "Admin", "User", "95%", "print('hi')")
    def mock_item(iid, option=None):
        if option == "values": return vals
        return {"values": vals}
    mock_tree.item.side_effect = mock_item

    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)
    monkeypatch.setattr(gptscan.messagebox, 'showinfo', MagicMock())

    sarif_path = tmp_path / "results.sarif"
    monkeypatch.setattr(gptscan.tkinter.filedialog, 'asksaveasfilename', lambda **k: str(sarif_path))

    gptscan.export_results()

    assert sarif_path.exists()
    content = sarif_path.read_text()
    assert '"version": "2.1.0"' in content
