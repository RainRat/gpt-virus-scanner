"""Tests for -l, --paths-only, --files-with-matches CLI functionality in gptscan."""

import os
import sys
from unittest.mock import MagicMock, patch
import pytest

import gptscan


def test_cli_paths_only_basic(capsys, monkeypatch):
    """Verify that run_cli with paths_only=True outputs only unique file paths."""
    mock_findings = [
        ('path/to/file1.py', '90%', 'Admin notes', 'User notes', '95%', 'eval(x)'),
        ('path/to/file2.py', '80%', 'Admin notes', 'User notes', '85%', 'exec(y)'),
    ]

    def mock_scan(*args, **kwargs):
        yield ('progress', (1, 2, 'Scanning...'))
        for item in mock_findings:
            yield ('result', item)
        yield ('summary', (2, 1024, 0.5))

    monkeypatch.setattr(gptscan, 'scan_files', mock_scan)

    result = gptscan.run_cli(
        targets=['.'],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        quiet=True,
        paths_only=True,
    )

    assert result == 2
    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
    assert lines == ['path/to/file1.py', 'path/to/file2.py']


def test_cli_paths_only_deduplication(capsys, monkeypatch):
    """Verify that multiple findings for the same file path are deduplicated."""
    mock_findings = [
        ('path/to/file1.py', '90%', 'Admin notes 1', 'User notes 1', '95%', 'eval(x)'),
        ('path/to/file1.py', '85%', 'Admin notes 2', 'User notes 2', '90%', 'exec(y)'),
        ('path/to/file2.py', '70%', 'Admin notes 3', 'User notes 3', '75%', 'system(z)'),
    ]

    def mock_scan(*args, **kwargs):
        yield ('progress', (1, 2, 'Scanning...'))
        for item in mock_findings:
            yield ('result', item)
        yield ('summary', (2, 1024, 0.5))

    monkeypatch.setattr(gptscan, 'scan_files', mock_scan)

    result = gptscan.run_cli(
        targets=['.'],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        quiet=True,
        paths_only=True,
    )

    assert result == 3
    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
    assert lines == ['path/to/file1.py', 'path/to/file2.py']


def test_cli_paths_only_sorting_and_top(capsys, monkeypatch):
    """Verify paths_only combined with sort_by and top_limit."""
    mock_findings = [
        ('path/to/low_threat.py', '55%', '', '', '55%', 'code1'),
        ('path/to/high_threat.py', '95%', '', '', '95%', 'code2'),
    ]

    def mock_scan(*args, **kwargs):
        for item in mock_findings:
            yield ('result', item)

    monkeypatch.setattr(gptscan, 'scan_files', mock_scan)

    gptscan.run_cli(
        targets=['.'],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        quiet=True,
        paths_only=True,
        sort_by='threat',
        top_limit=1,
    )

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
    assert lines == ['path/to/high_threat.py']


def test_cli_paths_only_file_output(tmp_path, monkeypatch):
    """Verify paths_only output saved to a file via output_file."""
    mock_findings = [
        ('path/to/script.py', '80%', '', '', '80%', 'import os'),
    ]

    def mock_scan(*args, **kwargs):
        for item in mock_findings:
            yield ('result', item)

    monkeypatch.setattr(gptscan, 'scan_files', mock_scan)

    out_file = str(tmp_path / "paths.txt")
    gptscan.run_cli(
        targets=['.'],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        quiet=True,
        paths_only=True,
        output_file=out_file,
    )

    content = (tmp_path / "paths.txt").read_text(encoding="utf-8")
    assert content.strip() == 'path/to/script.py'


@pytest.mark.parametrize("flag", ["-l", "--paths-only", "--files-with-matches"])
def test_cli_paths_only_flags_parsing(flag, monkeypatch):
    """Verify that argparse correctly sets args.paths_only for all flag aliases."""
    test_args = ["gptscan.py", "target_folder", flag, "--cli"]
    monkeypatch.setattr(sys, 'argv', test_args)

    run_cli_mock = MagicMock(return_value=0)
    monkeypatch.setattr(gptscan, 'run_cli', run_cli_mock)

    gptscan.main()

    assert run_cli_mock.called
    _, kwargs = run_cli_mock.call_args
    assert kwargs.get('paths_only') is True
