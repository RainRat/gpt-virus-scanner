import sys
import pytest
from unittest.mock import MagicMock, patch
import gptscan

def test_import_results_splitlist_and_errors(monkeypatch):
    """Verify import_results with string file_paths trigger splitlist and handles load errors."""
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, "tree", mock_tree, raising=False)

    # 1. Mock askopenfilenames to return a string (not list/tuple) to trigger splitlist branch
    monkeypatch.setattr(gptscan.filedialog, "askopenfilenames", lambda **k: "file1.json file2.json")

    # 2. Mock root.tk.splitlist
    mock_root = MagicMock()
    mock_root.tk.splitlist.return_value = ["file1.json", "file2.json"]
    monkeypatch.setattr(gptscan, "root", mock_root, raising=False)

    # 3. Mock askyesno to return False (do not append)
    monkeypatch.setattr(gptscan.messagebox, "askyesno", lambda *a, **k: False)

    # 4. Mock load_report_file to succeed on file1.json and fail on file2.json
    def mock_load_report_file(path):
        if "file1.json" in path:
            return [{"path": "a.py", "own_conf": "20%", "snippet": "print(1)"}]
        raise OSError("Disk read error")

    monkeypatch.setattr(gptscan, "load_report_file", mock_load_report_file)

    # 5. Capture showerror and _finalize_import
    mock_showerror = MagicMock()
    monkeypatch.setattr(gptscan.messagebox, "showerror", mock_showerror)

    finalized_calls = []
    def mock_finalize_import(data, source, append=False):
        finalized_calls.append((data, source, append))

    monkeypatch.setattr(gptscan, "_finalize_import", mock_finalize_import)

    # Run function
    gptscan.import_results()

    # Verify root.tk.splitlist was called
    mock_root.tk.splitlist.assert_called_once_with("file1.json file2.json")

    # Verify messagebox.showerror was called with the correct error
    mock_showerror.assert_called_once()
    args, kwargs = mock_showerror.call_args
    assert "Import Errors" in args[0]
    assert "file2.json: Disk read error" in args[1]

    # Verify _finalize_import was called with findings from file1.json
    assert len(finalized_calls) == 1
    data, source, append = finalized_calls[0]
    assert len(data) == 1
    assert data[0]["path"] == "a.py"
    assert source == "file1.json"
    assert append is False

def test_import_results_empty_no_error(monkeypatch):
    """Verify import_results handles case where askopenfilenames succeeds but no data is returned."""
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, "tree", mock_tree, raising=False)

    monkeypatch.setattr(gptscan.filedialog, "askopenfilenames", lambda **k: ["empty.json"])
    monkeypatch.setattr(gptscan, "load_report_file", lambda path: [])

    mock_showwarning = MagicMock()
    monkeypatch.setattr(gptscan.messagebox, "showwarning", mock_showwarning)

    gptscan.import_results()

    # Verify warning is shown
    mock_showwarning.assert_called_once_with("Import Warning", "No data found in the selected files.")

def test_import_results_many_files_appends(monkeypatch):
    """Verify import_results with > 3 files correctly names source and appends results."""
    mock_tree = MagicMock()
    # has_existing is True
    mock_tree.get_children.return_value = ["item1"]
    monkeypatch.setattr(gptscan, "tree", mock_tree, raising=False)

    # Mock askopenfilenames with 4 files
    monkeypatch.setattr(gptscan.filedialog, "askopenfilenames", lambda **k: ["f1.json", "f2.json", "f3.json", "f4.json"])
    # Mock askyesno to return True (user wants to append)
    monkeypatch.setattr(gptscan.messagebox, "askyesno", lambda *a, **k: True)
    # Mock load_report_file
    monkeypatch.setattr(gptscan, "load_report_file", lambda path: [{"path": f"path_from_{path}"}])

    finalized_calls = []
    def mock_finalize_import(data, source, append=False):
        finalized_calls.append((data, source, append))
    monkeypatch.setattr(gptscan, "_finalize_import", mock_finalize_import)

    gptscan.import_results()

    # Verify _finalize_import parameters
    assert len(finalized_calls) == 1
    data, source, append = finalized_calls[0]
    assert len(data) == 4
    assert source == "4 files"
    assert append is True

def test_run_cli_report_output(monkeypatch, capsys):
    """Verify run_cli with output_format='report' prints sorted console triage report."""
    def mock_scan_files(*args, **kwargs):
        yield ('progress', (0, 2, None))
        yield ('result', ("low_threat.py", "10%", "low", "low", "", "print('low')", "1"))
        yield ('result', ("high_threat.py", "80%", "high", "high", "95%", "print('high')", "2"))
        yield ('progress', (2, 2, None))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    # Call run_cli with report format
    gptscan.run_cli("/dummy", deep=False, show_all=True, use_gpt=True, rate_limit=60, output_format="report")

    captured = capsys.readouterr()
    stdout_output = captured.out

    # Verify that both results are in output
    assert "low_threat.py" in stdout_output
    assert "high_threat.py" in stdout_output
    assert "--- GPT SCAN - CONSOLE TRIAGE REPORT" in stdout_output

    # Verify order of findings (high_threat first because it has 95% AI Threat, whereas low_threat has 10% Local Threat)
    high_idx = stdout_output.find("high_threat.py")
    low_idx = stdout_output.find("low_threat.py")
    assert high_idx < low_idx, "high_threat.py should appear before low_threat.py in the report"

def test_run_cli_report_output_with_color(monkeypatch, capsys):
    """Verify run_cli with output_format='report' uses color when out_stream is tty."""
    def mock_scan_files(*args, **kwargs):
        yield ('result', ("file.py", "90%", "", "", "", "print()", "1"))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    # Use a custom stream object with isatty returning True to simulate a terminal
    class MockTTYStream:
        def __init__(self):
            self.content = []
        def write(self, s):
            self.content.append(s)
        def isatty(self):
            return True

    mock_stream = MockTTYStream()

    # Call run_cli with report format on our mock stream
    gptscan.run_cli("/dummy", deep=False, show_all=True, use_gpt=False, rate_limit=60, output_format="report", output_file=None)

    # Temporarily intercept stdout manually to pass stream or mock it
    with patch("sys.stdout", mock_stream):
        gptscan.run_cli("/dummy", deep=False, show_all=True, use_gpt=False, rate_limit=60, output_format="report")

    stdout_output = "".join(mock_stream.content)
    assert "\033[" in stdout_output, "Color ANSI escape sequences should be in report output"
