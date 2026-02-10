import pytest
from unittest.mock import MagicMock, patch
import gptscan
import json
import os

def test_scan_summary_gui(monkeypatch):
    """Test that finish_scan_state updates the status label with a summary."""
    mock_status_label = MagicMock()
    monkeypatch.setattr(gptscan, 'status_label', mock_status_label, raising=False)
    monkeypatch.setattr(gptscan, 'root', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'scan_button', MagicMock(), raising=False)
    monkeypatch.setattr(gptscan, 'cancel_button', MagicMock(), raising=False)

    # 1 suspicious file (singular)
    gptscan.finish_scan_state(total_scanned=10, threats_found=1)
    mock_status_label.config.assert_called_with(text="Scan complete: 10 files scanned, 1 suspicious file found.")

    # 5 suspicious files (plural)
    gptscan.finish_scan_state(total_scanned=20, threats_found=5)
    mock_status_label.config.assert_called_with(text="Scan complete: 20 files scanned, 5 suspicious files found.")

    # No args (Ready)
    gptscan.finish_scan_state()
    mock_status_label.config.assert_called_with(text="Ready")

def test_treeview_highlighting(monkeypatch):
    """Test that insert_tree_row applies the correct tags based on confidence."""
    mock_tree = MagicMock()
    mock_tree.column.return_value = {'width': 100}
    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)
    monkeypatch.setattr(gptscan, 'default_font_measure', lambda x: 10)

    # High risk (> 80%)
    # values: (path, own_conf, admin, user, gpt_conf, snippet)
    gptscan.insert_tree_row(("path1", "90%", "admin", "user", "", "snippet"))
    # The last call to insert should have tags=('high-risk',)
    _, kwargs = mock_tree.insert.call_args
    assert kwargs['tags'] == ('high-risk',)

    # Medium risk (> 50%)
    gptscan.insert_tree_row(("path2", "60%", "admin", "user", "", "snippet"))
    _, kwargs = mock_tree.insert.call_args
    assert kwargs['tags'] == ('medium-risk',)

    # Safe
    gptscan.insert_tree_row(("path3", "10%", "admin", "user", "0%", "snippet"))
    _, kwargs = mock_tree.insert.call_args
    assert kwargs['tags'] == ()

def test_export_multi_format(monkeypatch, tmp_path):
    """Test exporting results to different formats."""
    mock_tree = MagicMock()
    cols = ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet")
    mock_tree.__getitem__.return_value = cols
    mock_tree.get_children.return_value = ["item1"]
    mock_tree.item.return_value = {"values": ("test.py", "90%", "Admin Notes", "User Notes", "0%", "print('hi')")}
    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)
    monkeypatch.setattr(gptscan.messagebox, 'showinfo', MagicMock())

    # JSON Export
    json_path = tmp_path / "results.json"
    monkeypatch.setattr(gptscan.tkinter.filedialog, 'asksaveasfilename', lambda **k: str(json_path))
    gptscan.export_results()
    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)
        assert data[0]["path"] == "test.py"
        assert data[0]["own_conf"] == "90%"

    # HTML Export
    html_path = tmp_path / "results.html"
    monkeypatch.setattr(gptscan.tkinter.filedialog, 'asksaveasfilename', lambda **k: str(html_path))
    gptscan.export_results()
    assert html_path.exists()
    assert "GPT Scan Report" in html_path.read_text()

def test_cli_summary(capsys, monkeypatch):
    """Test that CLI output includes a final summary."""
    # We need to mock scan_files to yield a result and progress
    def mock_scan_files(*args, **kwargs):
        yield ('progress', (1, 1, "Done"))
        yield ('result', ("test.py", "90%", "Admin", "User", "", "Snippet"))

    monkeypatch.setattr(gptscan, 'scan_files', mock_scan_files)

    # Redirect sys.stdout to avoid printing to real stdout during test
    monkeypatch.setattr(gptscan.sys, 'stdout', MagicMock())

    gptscan.run_cli(["."], deep=False, show_all=False, use_gpt=False, rate_limit=60)

    captured = capsys.readouterr()
    assert "Scan complete: 1 files scanned, 1 suspicious file found." in captured.err
