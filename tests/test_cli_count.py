import io
import sys
import pytest
from unittest.mock import MagicMock
import gptscan


def test_run_cli_count_only(monkeypatch, tmp_path):
    """Test run_cli with count_only=True outputs only the integer count."""
    def mock_scan_files(*args, **kwargs):
        yield ('progress', (0, 2, "Scanning"))
        yield ('result', ("/path/test1.py", "90%", "Admin Note", "User Note", "85%", "import os; os.system('rm')", "1"))
        yield ('result', ("/path/test2.py", "80%", "Admin Note", "User Note", "80%", "import sys", "2"))
        yield ('progress', (2, 2, "Complete"))
        yield ('summary', (2, 200, 0.5))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    out_file = tmp_path / "count_output.txt"

    # Run scan with count_only=True
    count = gptscan.run_cli(
        targets="/path/test1.py",
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_format='csv',
        output_file=str(out_file),
        count_only=True
    )

    assert count == 2
    content = out_file.read_text(encoding="utf-8").strip()
    assert content == "2"


def test_run_cli_count_only_stdout(monkeypatch, capsys):
    """Test run_cli with count_only=True writing to stdout."""
    def mock_scan_files(*args, **kwargs):
        yield ('progress', (0, 1, "Scanning"))
        yield ('result', ("/path/test1.py", "90%", "Admin Note", "User Note", "85%", "import os", "1"))
        yield ('summary', (1, 100, 0.5))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    count = gptscan.run_cli(
        targets="/path/test1.py",
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_format='csv',
        count_only=True,
        quiet=True
    )

    captured = capsys.readouterr()
    stdout_val = captured.out.strip()
    assert stdout_val == "1"
    assert count == 1


def test_cli_main_count_flag(monkeypatch):
    """Test main() with --cli and --count flag."""
    mock_run_cli = MagicMock(return_value=1)
    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)

    test_args = [
        "gptscan.py",
        "/path/test.py",
        "--cli",
        "--count",
        "-o",
        "output.txt"
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    gptscan.main()

    assert mock_run_cli.called
    _, kwargs = mock_run_cli.call_args
    assert kwargs.get("count_only") is True


def test_cli_count_with_quiet_and_top(monkeypatch, tmp_path):
    """Test run_cli with count_only, quiet, and top limit."""
    def mock_scan_files(*args, **kwargs):
        yield ('result', ("/path/t1.py", "95%", "Admin Note", "User Note", "90%", "code1", "1"))
        yield ('result', ("/path/t2.py", "85%", "Admin Note", "User Note", "80%", "code2", "2"))
        yield ('summary', (2, 200, 0.5))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    out_file = tmp_path / "top_count.txt"

    count = gptscan.run_cli(
        targets=["/path/t1.py", "/path/t2.py"],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        quiet=True,
        top_limit=1,
        count_only=True,
        output_file=str(out_file)
    )

    content = out_file.read_text(encoding="utf-8").strip()
    assert content == "1"
