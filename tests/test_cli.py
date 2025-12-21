import csv
import io
import gptscan
import pytest

def test_run_cli_output_csv_format(monkeypatch, capsys):
    # Mock scan_files to yield predictable results
    def mock_scan_files(*args, **kwargs):
        yield ('progress', (0, 1, None))
        yield ('result', ("/path/file.py", "95%", "Admin Info", "User Info", "90%", "print('test')"))
        yield ('progress', (1, 1, None))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    # Run CLI
    gptscan.run_cli("/dummy/path", deep=False, show_all=True, use_gpt=True, rate_limit=60)

    # Capture output
    captured = capsys.readouterr()

    # Verify CSV output
    f = io.StringIO(captured.out)
    reader = csv.reader(f)
    rows = list(reader)

    assert len(rows) == 2
    assert rows[0] == ["path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet"]
    assert rows[1] == ["/path/file.py", "95%", "Admin Info", "User Info", "90%", "print('test')"]

def test_run_cli_handles_special_characters(monkeypatch, capsys):
    # Mock scan_files with special characters in snippet
    snippet_with_chars = 'line1, "quoted", \nline2'

    def mock_scan_files(*args, **kwargs):
        yield ('result', ("/path/complex.py", "50%", "", "", "", snippet_with_chars))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    gptscan.run_cli("/dummy", False, True, False, 60)

    captured = capsys.readouterr()
    f = io.StringIO(captured.out)
    reader = csv.reader(f)
    rows = list(reader)

    # Row 1 is header, Row 2 is data
    assert rows[1][5] == snippet_with_chars
