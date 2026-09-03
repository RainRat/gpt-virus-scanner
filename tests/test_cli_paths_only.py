import sys
import pytest
from unittest.mock import MagicMock
import gptscan


def test_run_cli_paths_only(monkeypatch, tmp_path):
    """Test run_cli with paths_only=True outputs unique file paths line-by-line."""
    def mock_scan_files(*args, **kwargs):
        yield ('progress', (0, 3, "Scanning"))
        yield ('result', ("/path/test1.py", "90%", "Admin Note", "User Note", "85%", "import os; os.system('rm')", "1"))
        yield ('result', ("/path/test1.py", "95%", "Admin Note", "User Note", "90%", "eval('x')", "10"))
        yield ('result', ("/path/test2.py", "80%", "Admin Note", "User Note", "80%", "import sys", "2"))
        yield ('progress', (3, 3, "Complete"))
        yield ('summary', (3, 300, 0.5))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    out_file = tmp_path / "paths_output.txt"

    threats = gptscan.run_cli(
        targets="/path/test1.py",
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_format='csv',
        output_file=str(out_file),
        paths_only=True
    )

    assert threats == 3
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert lines == ["/path/test1.py", "/path/test2.py"]


def test_run_cli_paths_only_stdout(monkeypatch, capsys):
    """Test run_cli with paths_only=True writing to stdout."""
    def mock_scan_files(*args, **kwargs):
        yield ('progress', (0, 1, "Scanning"))
        yield ('result', ("/path/test1.py", "90%", "Admin Note", "User Note", "85%", "import os", "1"))
        yield ('summary', (1, 100, 0.5))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    threats = gptscan.run_cli(
        targets="/path/test1.py",
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_format='csv',
        paths_only=True,
        quiet=True
    )

    captured = capsys.readouterr()
    stdout_lines = captured.out.strip().splitlines()
    assert stdout_lines == ["/path/test1.py"]
    assert threats == 1


@pytest.mark.parametrize("flag", ["--paths-only", "-l", "--files-with-matches"])
def test_cli_main_paths_only_flags(monkeypatch, flag):
    """Test main() with --cli and path-only flags (-l, --paths-only, --files-with-matches)."""
    mock_run_cli = MagicMock(return_value=1)
    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)

    test_args = [
        "gptscan.py",
        "/path/test.py",
        "--cli",
        flag,
        "-o",
        "output.txt"
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    gptscan.main()

    assert mock_run_cli.called
    _, kwargs = mock_run_cli.call_args
    assert kwargs.get("paths_only") is True


def test_run_cli_paths_only_with_sorting_and_top(monkeypatch, tmp_path):
    """Test run_cli with paths_only, sort_by='path', and top_limit."""
    def mock_scan_files(*args, **kwargs):
        yield ('result', ("/path/z_file.py", "95%", "Admin Note", "User Note", "90%", "code1", "1"))
        yield ('result', ("/path/a_file.py", "85%", "Admin Note", "User Note", "80%", "code2", "2"))
        yield ('summary', (2, 200, 0.5))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    out_file = tmp_path / "sorted_paths.txt"

    threats = gptscan.run_cli(
        targets=["/path/z_file.py", "/path/a_file.py"],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        quiet=True,
        sort_by="path",
        paths_only=True,
        output_file=str(out_file)
    )

    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert lines == ["/path/a_file.py", "/path/z_file.py"]
    assert threats == 2


def test_run_cli_paths_only_empty(monkeypatch, tmp_path):
    """Test run_cli with paths_only=True when no threats are found."""
    def mock_scan_files(*args, **kwargs):
        yield ('summary', (1, 100, 0.5))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    out_file = tmp_path / "empty_paths.txt"

    threats = gptscan.run_cli(
        targets="/path/clean.py",
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        quiet=True,
        paths_only=True,
        output_file=str(out_file)
    )

    assert threats == 0
    content = out_file.read_text(encoding="utf-8")
    assert content == ""
