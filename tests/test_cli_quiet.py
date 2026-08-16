import sys
import json
from unittest.mock import MagicMock
import pytest
import gptscan


def test_run_cli_quiet_mode_suppresses_stderr(monkeypatch, capsys):
    """Verify run_cli with quiet=True suppresses progress and summary on sys.stderr."""
    def mock_scan_files(*args, **kwargs):
        yield ('progress', (0, 1, "Scanning"))
        yield ('result', ("/path/test.py", "90%", "Admin Note", "User Note", "85%", "import os; os.system('rm -rf /')", "1"))
        yield ('progress', (1, 1, "Complete"))
        yield ('summary', (1, 100, 0.5))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    # Call run_cli with quiet=True and json output format
    threats = gptscan.run_cli(
        targets=["/path/test.py"],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        quiet=True
    )

    captured = capsys.readouterr()
    # stderr should be completely empty in quiet mode
    assert captured.err == ""
    # stdout should contain json output
    assert "/path/test.py" in captured.out
    assert threats == 1


def test_run_cli_quiet_mode_false_emits_stderr(monkeypatch, capsys):
    """Verify run_cli with quiet=False emits progress and summary on sys.stderr."""
    def mock_scan_files(*args, **kwargs):
        yield ('progress', (0, 1, "Scanning"))
        yield ('result', ("/path/test.py", "90%", "Admin Note", "User Note", "85%", "import os; os.system('rm -rf /')", "1"))
        yield ('progress', (1, 1, "Complete"))
        yield ('summary', (1, 100, 0.5))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    threats = gptscan.run_cli(
        targets=["/path/test.py"],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        quiet=False
    )

    captured = capsys.readouterr()
    # stderr should contain progress/summary information
    assert "scanned" in captured.err or "Scanning" in captured.err or "files" in captured.err
    # stdout should contain json output
    assert "/path/test.py" in captured.out
    assert threats == 1


def test_main_quiet_flag_parsing(monkeypatch):
    """Verify main() parses -q and --quiet flags and passes quiet=True to run_cli."""
    mock_run_cli = MagicMock(return_value=0)
    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)

    test_args = ["gptscan.py", "--cli", "-q", "."]
    monkeypatch.setattr(sys, "argv", test_args)

    gptscan.main()

    assert mock_run_cli.called
    _, kwargs = mock_run_cli.call_args
    assert kwargs.get("quiet") is True
