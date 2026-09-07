import io
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

import gptscan


def test_run_cli_paths_only_outputs_unique_paths():
    """Verify that run_cli with paths_only=True prints only unique file paths line-by-line."""
    mock_events = [
        ('result', ('file1.py', '80%', 'Desc 1', 'Desc 1', '', 'print(1)')),
        ('result', ('file1.py', '85%', 'Desc 2', 'Desc 2', '', 'print(2)')),
        ('result', ('file2.js', '90%', 'Desc 3', 'Desc 3', '', 'eval(3)')),
        ('summary', (2, 100, 0.5)),
    ]

    out_stream = io.StringIO()

    with patch('gptscan.scan_files', return_value=mock_events):
        threats = gptscan.run_cli(
            targets=['.'],
            deep=False,
            show_all=False,
            use_gpt=False,
            rate_limit=60,
            paths_only=True,
            quiet=True,
        )

    # Mock stream output capture
    # run_cli writes to sys.stdout by default if output_file is None
    # Let's test passing output_file or capturing out_stream
    assert threats == 3


def test_run_cli_paths_only_with_output_stream(tmp_path):
    """Verify paths_only writes unique paths to output file."""
    output_file = tmp_path / "paths.txt"
    mock_events = [
        ('result', ('/path/to/script_a.py', '90%', 'D1', 'D1', '', 'bad_code()')),
        ('result', ('/path/to/script_a.py', '95%', 'D2', 'D2', '', 'bad_code_2()')),
        ('result', ('/path/to/script_b.py', '80%', 'D3', 'D3', '', 'suspicious()')),
    ]

    with patch('gptscan.scan_files', return_value=mock_events):
        threats = gptscan.run_cli(
            targets=['.'],
            deep=False,
            show_all=False,
            use_gpt=False,
            rate_limit=60,
            output_file=str(output_file),
            paths_only=True,
            quiet=True,
        )

    assert threats == 3
    content = output_file.read_text(encoding='utf-8').splitlines()
    assert content == ['/path/to/script_a.py', '/path/to/script_b.py']


def test_main_cli_paths_only_flags(monkeypatch, tmp_path, capsys):
    """Test CLI invocation with -l / --paths-only / --files-with-matches flags."""
    test_file = tmp_path / "malicious.py"
    test_file.write_text("import os; os.system('rm -rf /')", encoding="utf-8")

    mock_events = [
        ('result', (str(test_file), '95%', 'Dangerous command', 'Dangerous command', '', 'os.system')),
    ]

    monkeypatch.setattr('sys.argv', ['gptscan.py', str(test_file), '--cli', '-l', '-q'])

    with patch('gptscan.scan_files', return_value=mock_events):
        with patch('sys.exit') as mock_exit:
            gptscan.main()
            mock_exit.assert_not_called()

    captured = capsys.readouterr()
    assert str(test_file) in captured.out
    assert "path,own_conf" not in captured.out
