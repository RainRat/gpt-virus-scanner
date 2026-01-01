import sys
import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

# We need to ensure we can import gptscan even if it's not installed
sys.path.append(str(Path(__file__).parent.parent))

# Import gptscan module. Since it has global side effects (Config.initialize),
# it might print stuff, but we can capture it.
import gptscan

def test_cli_file_list_explicit_file(tmp_path, capsys):
    """Test reading scan targets from a file specified by --file-list."""
    # Create dummy files to scan
    f1 = tmp_path / "file1.py"
    f1.touch()
    f2 = tmp_path / "file2.py"
    f2.touch()

    # Create the list file
    list_file = tmp_path / "targets.txt"
    list_file.write_text(f"{f1}\n{f2}\n")

    # Mock sys.argv
    test_args = ["gptscan.py", "--cli", "--file-list", str(list_file), "--dry-run"]

    # Mock run_cli to verify it receives the correct targets
    with patch("gptscan.run_cli") as mock_run_cli:
        with patch.object(sys, 'argv', test_args):
            gptscan.main()

            mock_run_cli.assert_called_once()
            args, kwargs = mock_run_cli.call_args
            targets = args[0]

            # Verify both files are in the targets
            assert str(f1) in targets
            assert str(f2) in targets

def test_cli_file_list_stdin(tmp_path, capsys):
    """Test reading scan targets from stdin using --file-list -."""
    f1 = tmp_path / "stdin_file.py"
    f1.touch()

    # Content to be read from stdin
    stdin_lines = [f"{f1}\n"]

    test_args = ["gptscan.py", "--cli", "--file-list", "-", "--dry-run"]

    with patch("gptscan.run_cli") as mock_run_cli:
        with patch.object(sys, 'argv', test_args):
            # Create a mock for stdin that behaves like a file object (iterable)
            mock_stdin = MagicMock()
            mock_stdin.__iter__.return_value = iter(stdin_lines)

            with patch('sys.stdin', mock_stdin):
                gptscan.main()

                mock_run_cli.assert_called_once()
                args, kwargs = mock_run_cli.call_args
                targets = args[0]

                assert str(f1) in targets
