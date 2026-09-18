import os
import sys
import subprocess
from unittest.mock import patch, MagicMock
import pytest

import gptscan
from gptscan import get_git_staged_files, scan_git_staged_click


def test_get_git_staged_files_error_handling():
    """Test behavior when git command fails (e.g. not a repo or git not installed)."""
    with patch("subprocess.check_output") as mock_check_output:
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "git")

        files = get_git_staged_files()
        assert files == []
        assert mock_check_output.call_count == 1


def test_get_git_staged_files_no_staged_files():
    """Test when git reports no staged files."""
    with patch("subprocess.check_output") as mock_check_output:
        # rev-parse returns empty/toplevel, git diff --cached returns empty
        mock_check_output.side_effect = ["", ""]

        files = get_git_staged_files()
        assert files == []
        assert mock_check_output.call_count == 2


def test_get_git_staged_files_mocked_success():
    """Test detecting staged files with mocked git output."""
    toplevel = os.getcwd()
    with patch("subprocess.check_output") as mock_check_output, \
         patch("os.path.exists", return_value=True):

        mock_check_output.side_effect = [
            toplevel,
            "staged1.py\nstaged2.py\n"
        ]

        files = get_git_staged_files()
        assert len(files) == 2
        expected = [os.path.join(toplevel, "staged1.py"), os.path.join(toplevel, "staged2.py")]
        assert files == expected


def test_get_git_staged_files_real_git(tmp_path):
    """Verify that staged files are correctly detected in a real git repository."""
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(tmp_path), check=True)

    staged_file = tmp_path / "staged.py"
    staged_file.write_text("print('staged')")

    unstaged_file = tmp_path / "unstaged.py"
    unstaged_file.write_text("print('unstaged')")
    # Commit unstaged_file first so it can be modified without staging
    subprocess.run(["git", "add", "unstaged.py"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True)
    unstaged_file.write_text("print('unstaged modified')")

    untracked_file = tmp_path / "untracked.py"
    untracked_file.write_text("print('untracked')")

    # Only stage staged_file
    subprocess.run(["git", "add", "staged.py"], cwd=str(tmp_path), check=True)

    results = get_git_staged_files(str(tmp_path))
    result_paths = [os.path.abspath(r) for r in results]

    assert os.path.abspath(staged_file) in result_paths
    assert os.path.abspath(unstaged_file) not in result_paths
    assert os.path.abspath(untracked_file) not in result_paths


def test_scan_git_staged_click(mocker):
    """Test that scan_git_staged_click calls _generic_scan_click properly."""
    mock_generic = mocker.patch("gptscan._generic_scan_click")
    scan_git_staged_click()
    assert mock_generic.call_count == 1
    args, kwargs = mock_generic.call_args
    assert args[1] == "Git Staged Files"
    assert args[2] == "No staged Git files found in the target path."
    assert args[3] == "Git Staged Files Error"


def test_cli_git_staged_flag(tmp_path, mocker):
    """Test running CLI with --git-staged."""
    staged = tmp_path / "staged.py"
    staged.write_text("print('test')")

    scan_called_targets = []

    def mock_run_cli(scan_targets, *args, **kwargs):
        scan_called_targets.extend(scan_targets)
        return 0

    mocker.patch("gptscan.run_cli", mock_run_cli)
    mocker.patch("gptscan.get_git_staged_files", return_value=[str(staged)])

    with patch("sys.argv", ["gptscan.py", str(tmp_path), "--git-staged", "--cli"]):
        gptscan.main()

    assert scan_called_targets == [str(staged)]
