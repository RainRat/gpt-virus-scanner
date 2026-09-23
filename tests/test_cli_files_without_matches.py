import io
import pytest
from unittest.mock import MagicMock, patch
import gptscan


def test_cli_files_without_matches_flag(tmp_path, monkeypatch):
    """Test -L / --files-without-matches outputs clean scanned files and omits suspicious ones."""
    clean_file = tmp_path / "safe.py"
    clean_file.write_text("print('hello world')", encoding="utf-8")

    suspicious_file = tmp_path / "bad.py"
    suspicious_file.write_text("import os; os.system('rm -rf /')", encoding="utf-8")

    # Mock scan_files generator output
    # (path, own_conf, admin, user, gpt, snippet, line)
    def mock_scan_files(*args, **kwargs):
        yield ('progress', (1, 2, "Scanning safe.py"))
        yield ('result', (str(clean_file), "0%", "", "", "", "print('hello world')", 1))
        yield ('progress', (2, 2, "Scanning bad.py"))
        yield ('result', (str(suspicious_file), "95%", "High threat", "User danger", "95%", "os.system('rm -rf /')", 1))
        yield ('summary', (2, 100, 0.5))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    out_stream = io.StringIO()
    with patch("sys.stdout", out_stream):
        threats = gptscan.run_cli(
            targets=[str(tmp_path)],
            deep=False,
            show_all=False,
            use_gpt=False,
            rate_limit=60,
            output_format='csv',
            files_without_matches=True,
            quiet=True
        )

    output = out_stream.getvalue().splitlines()
    assert str(clean_file) in output
    assert str(suspicious_file) not in output
    assert threats == 1


def test_cli_files_without_matches_output_file(tmp_path, monkeypatch):
    """Test -L / --files-without-matches when writing output to a file via output_file."""
    clean_file = tmp_path / "safe.py"
    clean_file.write_text("a = 1", encoding="utf-8")

    out_file = tmp_path / "clean_list.txt"

    def mock_scan_files(*args, **kwargs):
        yield ('result', (str(clean_file), "0%", "", "", "", "a = 1", 1))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    gptscan.run_cli(
        targets=[str(clean_file)],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_file=str(out_file),
        files_without_matches=True,
        quiet=True
    )

    content = out_file.read_text(encoding="utf-8").splitlines()
    assert str(clean_file) in content


def test_cli_files_without_matches_no_clean_files(tmp_path, monkeypatch):
    """Test -L / --files-without-matches when all scanned files are threats."""
    suspicious_file = tmp_path / "bad.py"

    def mock_scan_files(*args, **kwargs):
        yield ('result', (str(suspicious_file), "90%", "Danger", "User warning", "", "eval(x)", 1))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    out_stream = io.StringIO()
    with patch("sys.stdout", out_stream):
        gptscan.run_cli(
            targets=[str(suspicious_file)],
            deep=False,
            show_all=False,
            use_gpt=False,
            rate_limit=60,
            files_without_matches=True,
            quiet=True
        )

    output = out_stream.getvalue().strip()
    assert output == ""
