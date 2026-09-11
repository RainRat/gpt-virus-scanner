import json
import pytest
from unittest.mock import MagicMock
import gptscan


def test_import_results_generator_clipboard_json(monkeypatch):
    """Test importing scan results from clipboard via import_results_generator."""
    report_json = json.dumps([
        {
            "path": "test_script.py",
            "line": "12",
            "own_conf": "75%",
            "gpt_conf": "85%",
            "admin_desc": "Suspicious eval execution",
            "end-user_desc": "Dangerous code",
            "snippet": "eval('malicious')"
        }
    ])
    monkeypatch.setattr(gptscan, "get_cli_clipboard_content", lambda: report_json)

    events = list(gptscan.import_results_generator("clipboard"))

    result_events = [e for e in events if e[0] == 'result']
    assert len(result_events) == 1
    data = result_events[0][1]
    assert data[0] == "test_script.py"
    assert data[1] == "75%"
    assert data[2] == "Suspicious eval execution"
    assert data[3] == "Dangerous code"
    assert data[4] == "85%"
    assert data[5] == "eval('malicious')"
    assert data[6] == "12"


def test_import_results_generator_clipboard_empty(monkeypatch):
    """Test importing scan results when clipboard is empty."""
    monkeypatch.setattr(gptscan, "get_cli_clipboard_content", lambda: None)

    events = list(gptscan.import_results_generator("clipboard"))

    progress_events = [e for e in events if e[0] == 'progress']
    assert len(progress_events) == 1
    assert "Clipboard content is empty or unavailable" in progress_events[0][1][2]


def test_load_report_file_clipboard_json(monkeypatch):
    """Test load_report_file when file_path is 'clipboard'."""
    report_json = json.dumps([
        {
            "path": "clipboard_test.py",
            "line": "5",
            "own_conf": "90%",
            "gpt_conf": "95%",
            "admin_desc": "Admin note",
            "end-user_desc": "User note",
            "snippet": "import os; os.system('ls')"
        }
    ])
    monkeypatch.setattr(gptscan, "get_cli_clipboard_content", lambda: report_json)

    items = gptscan.load_report_file("clipboard")
    assert len(items) == 1
    assert items[0]["path"] == "clipboard_test.py"
    assert items[0]["own_conf"] == "90%"


def test_load_report_file_clipboard_empty_raises(monkeypatch):
    """Test load_report_file raises ValueError when clipboard is empty."""
    monkeypatch.setattr(gptscan, "get_cli_clipboard_content", lambda: "")

    with pytest.raises(ValueError, match="Clipboard content is empty or unavailable"):
        gptscan.load_report_file("clipboard")


def test_run_cli_import_results_clipboard(monkeypatch, capsys):
    """Test running CLI with --import-results clipboard."""
    report_json = json.dumps([
        {
            "path": "imported_via_cli.py",
            "line": "1",
            "own_conf": "80%",
            "gpt_conf": "80%",
            "admin_desc": "Test admin",
            "end-user_desc": "Test user",
            "snippet": "exec(code)"
        }
    ])
    monkeypatch.setattr(gptscan, "get_cli_clipboard_content", lambda: report_json)

    threats = gptscan.run_cli(
        targets=[],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        import_file="clipboard"
    )

    captured = capsys.readouterr()
    assert threats == 1
    assert "imported_via_cli.py" in captured.out


def test_run_cli_baseline_clipboard(monkeypatch, tmp_path, capsys):
    """Test running CLI with --baseline clipboard to filter existing findings."""
    baseline_json = json.dumps([
        {
            "path": "known_issue.py",
            "line": "10",
            "own_conf": "90%",
            "gpt_conf": "90%",
            "snippet": "os.system('id')"
        }
    ])
    monkeypatch.setattr(gptscan, "get_cli_clipboard_content", lambda: baseline_json)

    # Mock scan generator yielding one baseline match and one new finding
    def mock_scan_files(*args, **kwargs):
        yield ('result', ("known_issue.py", "90%", "Admin", "User", "90%", "os.system('id')", "10"))
        yield ('result', ("new_issue.py", "95%", "Admin2", "User2", "95%", "os.system('rm')", "20"))
        yield ('summary', (2, 2, 0.1))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    threats = gptscan.run_cli(
        targets=[str(tmp_path)],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        baseline_file="clipboard"
    )

    captured = capsys.readouterr()
    assert threats == 1
    assert "new_issue.py" in captured.out
    assert "known_issue.py" not in captured.out
