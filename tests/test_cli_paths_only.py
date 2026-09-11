import sys
from unittest.mock import MagicMock
import pytest
import gptscan


def test_run_cli_paths_only_output(capsys, monkeypatch):
    """Verify that run_cli with paths_only=True outputs unique file paths line-by-line without headers."""
    sample_results = [
        ('result', ('script1.py', '90%', 'Admin note 1', 'User note 1', '90%', 'exec(bad)', '10')),
        ('result', ('script1.py', '85%', 'Admin note 2', 'User note 2', '85%', 'eval(bad)', '25')),
        ('result', ('script2.py', '70%', 'Admin note 3', 'User note 3', '70%', 'system(bad)', '5')),
    ]

    monkeypatch.setattr(gptscan, 'scan_files', lambda *args, **kwargs: sample_results)

    threats = gptscan.run_cli(
        targets=['.'],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        quiet=True,
        paths_only=True
    )

    captured = capsys.readouterr()
    output_lines = captured.out.strip().splitlines()

    assert threats == 3
    assert output_lines == ['script1.py', 'script2.py']


def test_main_cli_paths_only_flag(capsys, monkeypatch):
    """Verify that passing -l / --paths-only / --files-with-matches to main works correctly."""
    sample_results = [
        ('result', ('/path/to/bad1.py', '95%', 'Desc', 'Desc', '95%', 'snippet', '1')),
        ('result', ('/path/to/bad2.py', '80%', 'Desc', 'Desc', '80%', 'snippet', '2')),
    ]

    monkeypatch.setattr(gptscan, 'scan_files', lambda *args, **kwargs: sample_results)

    for flag in ['-l', '--paths-only', '--files-with-matches']:
        monkeypatch.setattr(sys, 'argv', ['gptscan.py', './test_folder', '--cli', '-q', flag])

        gptscan.main()
        captured = capsys.readouterr()
        output_lines = captured.out.strip().splitlines()

        assert output_lines == ['/path/to/bad1.py', '/path/to/bad2.py']


def test_run_cli_paths_only_with_output_file(tmp_path, monkeypatch):
    """Verify that paths_only outputs to a specified output file."""
    sample_results = [
        ('result', ('fileA.py', '80%', 'A', 'A', '80%', 'snip', '1')),
        ('result', ('fileB.py', '85%', 'B', 'B', '85%', 'snip', '2')),
    ]

    monkeypatch.setattr(gptscan, 'scan_files', lambda *args, **kwargs: sample_results)

    out_file = tmp_path / "paths.txt"

    threats = gptscan.run_cli(
        targets=['.'],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_file=str(out_file),
        quiet=True,
        paths_only=True
    )

    assert threats == 2
    content = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert content == ['fileA.py', 'fileB.py']
