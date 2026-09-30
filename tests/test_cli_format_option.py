import os
import sys
from unittest.mock import patch, MagicMock
import pytest
import gptscan


def test_cli_format_short_flag(monkeypatch, tmp_path):
    """Test that -f short flag sets the output format correctly in run_cli."""
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')", encoding="utf-8")

    mock_run_cli = MagicMock(return_value=0)
    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)

    test_args = ["gptscan.py", str(test_file), "--cli", "-f", "json"]
    monkeypatch.setattr(sys, "argv", test_args)

    gptscan.main()

    mock_run_cli.assert_called_once()
    _, kwargs = mock_run_cli.call_args
    assert kwargs["output_format"] == "json"


def test_cli_format_long_flag_and_aliases(monkeypatch, tmp_path):
    """Test --format long flag and format aliases (jsonl -> ndjson, md -> markdown, yml -> yaml)."""
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')", encoding="utf-8")

    formats_and_expected = [
        ("json", "json"),
        ("ndjson", "ndjson"),
        ("jsonl", "ndjson"),
        ("csv", "csv"),
        ("tsv", "tsv"),
        ("sarif", "sarif"),
        ("html", "html"),
        ("md", "markdown"),
        ("markdown", "markdown"),
        ("xml", "xml"),
        ("junit", "junit"),
        ("yaml", "yaml"),
        ("yml", "yaml"),
        ("report", "report"),
    ]

    for fmt_input, expected_fmt in formats_and_expected:
        mock_run_cli = MagicMock(return_value=0)
        monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)

        test_args = ["gptscan.py", str(test_file), "--cli", "--format", fmt_input]
        monkeypatch.setattr(sys, "argv", test_args)

        gptscan.main()

        mock_run_cli.assert_called_once()
        _, kwargs = mock_run_cli.call_args
        assert kwargs["output_format"] == expected_fmt, f"Failed for format input: {fmt_input}"


def test_cli_format_case_insensitivity(monkeypatch, tmp_path):
    """Test that --format option is case-insensitive."""
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')", encoding="utf-8")

    mock_run_cli = MagicMock(return_value=0)
    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)

    test_args = ["gptscan.py", str(test_file), "--cli", "--format", "SARIF"]
    monkeypatch.setattr(sys, "argv", test_args)

    gptscan.main()

    mock_run_cli.assert_called_once()
    _, kwargs = mock_run_cli.call_args
    assert kwargs["output_format"] == "sarif"


def test_cli_format_invalid_choice(monkeypatch, tmp_path, capsys):
    """Test that specifying an invalid format causes argparse to exit with an error."""
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')", encoding="utf-8")

    test_args = ["gptscan.py", str(test_file), "--cli", "--format", "invalid_format"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        gptscan.main()

    assert exc_info.value.code != 0


def test_cli_format_with_output_file(monkeypatch, tmp_path):
    """Test that --format overrides extension-inferred format when both are supplied."""
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')", encoding="utf-8")
    out_file = tmp_path / "report.txt"

    mock_run_cli = MagicMock(return_value=0)
    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)

    test_args = ["gptscan.py", str(test_file), "--cli", "--format", "json", "-o", str(out_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    gptscan.main()

    mock_run_cli.assert_called_once()
    _, kwargs = mock_run_cli.call_args
    assert kwargs["output_format"] == "json"
    assert kwargs["output_file"] == str(out_file)
