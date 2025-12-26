import json
import io
import gptscan
import pytest

def test_run_cli_output_json_format(monkeypatch, capsys):
    # Mock scan_files to yield predictable results
    def mock_scan_files(*args, **kwargs):
        yield ('progress', (0, 1, None))
        yield ('result', ("/path/file.py", "95%", "Admin Info", "User Info", "90%", "print('test')"))
        yield ('progress', (1, 1, None))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    # Run CLI with json output
    gptscan.run_cli("/dummy/path", deep=False, show_all=True, use_gpt=True, rate_limit=60, output_format='json')

    # Capture output
    captured = capsys.readouterr()

    # Verify JSON output
    lines = captured.out.strip().splitlines()
    assert len(lines) == 1

    data = json.loads(lines[0])
    expected = {
        "path": "/path/file.py",
        "own_conf": "95%",
        "admin_desc": "Admin Info",
        "end-user_desc": "User Info",
        "gpt_conf": "90%",
        "snippet": "print('test')"
    }
    assert data == expected

def test_run_cli_json_handles_special_characters(monkeypatch, capsys):
    snippet_with_chars = 'line1, "quoted", \nline2'

    def mock_scan_files(*args, **kwargs):
        yield ('result', ("/path/complex.py", "50%", "", "", "", snippet_with_chars))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    gptscan.run_cli("/dummy", False, True, False, 60, output_format='json')

    captured = capsys.readouterr()
    lines = captured.out.strip().splitlines()
    data = json.loads(lines[0])

    assert data["snippet"] == snippet_with_chars
