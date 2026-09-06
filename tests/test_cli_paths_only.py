import os
import sys
import pytest
from unittest.mock import MagicMock, patch

import gptscan


def test_run_cli_paths_only_output(capsys):
    """Test that run_cli with paths_only=True outputs unique file paths line-by-line without headers."""
    def mock_event_gen(*args, **kwargs):
        yield ('result', ('file1.py', '90%', 'Admin note 1', 'User note 1', '', 'os.system("rm")', 5))
        yield ('result', ('file1.py', '85%', 'Admin note 2', 'User note 2', '', 'eval("bad")', 12))
        yield ('result', ('file2.js', '95%', 'Admin note 3', 'User note 3', '', 'exec("bad")', 1))
        yield ('summary', (2, 100, 0.5))

    with patch.object(gptscan, 'scan_files', side_effect=mock_event_gen):
        threats = gptscan.run_cli(
            targets=['.'],
            deep=False,
            show_all=False,
            use_gpt=False,
            rate_limit=60,
            output_format='csv',
            paths_only=True,
            quiet=True
        )

    assert threats == 3
    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
    assert lines == ['file1.py', 'file2.js']


def test_run_cli_paths_only_to_file(tmp_path):
    """Test that run_cli with paths_only=True writes unique file paths to an output file."""
    output_file = tmp_path / "paths.txt"

    def mock_event_gen(*args, **kwargs):
        yield ('result', ('src/a.py', '90%', '', '', '', 'snippet1', 10))
        yield ('result', ('src/b.py', '80%', '', '', '', 'snippet2', 20))
        yield ('result', ('src/a.py', '95%', '', '', '', 'snippet3', 30))
        yield ('summary', (2, 200, 0.2))

    with patch.object(gptscan, 'scan_files', side_effect=mock_event_gen):
        threats = gptscan.run_cli(
            targets=['.'],
            deep=False,
            show_all=False,
            use_gpt=False,
            rate_limit=60,
            output_file=str(output_file),
            paths_only=True,
            quiet=True
        )

    assert threats == 3
    lines = [line.strip() for line in output_file.read_text(encoding='utf-8').strip().splitlines() if line.strip()]
    assert lines == ['src/a.py', 'src/b.py']


def test_main_paths_only_aliases(monkeypatch, capsys):
    """Test CLI argument parsing for -l, --paths-only, and --files-with-matches."""
    for flag in ['-l', '--paths-only', '--files-with-matches']:
        test_args = ['gptscan.py', 'target.py', '--cli', '-q', flag]
        monkeypatch.setattr(sys, 'argv', test_args)

        mock_run_cli = MagicMock(return_value=0)
        monkeypatch.setattr(gptscan, 'run_cli', mock_run_cli)

        gptscan.main()

        assert mock_run_cli.called
        _, kwargs = mock_run_cli.call_args
        assert kwargs.get('paths_only') is True
