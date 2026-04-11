import sys
import io
from unittest.mock import MagicMock
import pytest
import gptscan

def test_run_cli_import_from_file(monkeypatch, capsys):
    mock_gen = MagicMock(return_value=iter([
        ('progress', (0, 1, "Loading")),
        ('result', ("imported.py", "80%", "Admin", "User", "85%", "code", "1")),
        ('summary', (1, 0, 0.0))
    ]))
    monkeypatch.setattr(gptscan, "import_results_generator", mock_gen)

    exit_code = gptscan.run_cli(
        targets=[],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        import_file="old_results.csv"
    )

    assert exit_code == 1
    mock_gen.assert_called_once_with("old_results.csv")
    captured = capsys.readouterr()
    assert "imported.py" in captured.out
    assert "80%" in captured.out

def test_run_cli_import_from_stdin(monkeypatch, capsys):
    mock_gen = MagicMock(return_value=iter([
        ('result', ("stdin.py", "90%", "", "", "95%", "print()", "10"))
    ]))
    monkeypatch.setattr(gptscan, "import_results_from_content_generator", mock_gen)

    monkeypatch.setattr(sys.stdin, "read", lambda: "some content")

    exit_code = gptscan.run_cli(
        targets=[],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        import_file="-"
    )

    assert exit_code == 1
    mock_gen.assert_called_once_with("some content")
    captured = capsys.readouterr()
    assert "stdin.py" in captured.out

def test_run_cli_import_from_stdin_error(monkeypatch, capsys):
    def mock_read_error():
        raise RuntimeError("Stdin failure")

    monkeypatch.setattr(sys.stdin, "read", mock_read_error)

    exit_code = gptscan.run_cli(
        targets=[],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        import_file="-"
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Stdin failure" in captured.err
