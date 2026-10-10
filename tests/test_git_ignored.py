import os
import sys
import subprocess
from unittest.mock import patch, MagicMock
import pytest

import gptscan
from gptscan import get_git_ignored_files, scan_git_ignored_click


def test_get_git_ignored_files_error_handling():
    """Test behavior when git command fails (e.g. not a repo or git not installed)."""
    with patch("subprocess.check_output") as mock_check_output:
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "git")

        files = get_git_ignored_files()
        assert files == []
        assert mock_check_output.call_count == 1


def test_get_git_ignored_files_subprocess_error_after_rev_parse():
    toplevel = os.getcwd()
    with patch("subprocess.check_output") as mock_check_output:
        mock_check_output.side_effect = [
            toplevel,
            subprocess.CalledProcessError(1, "git ls-files"),
        ]
        assert get_git_ignored_files() == []

    with patch("subprocess.check_output") as mock_check_output:
        mock_check_output.side_effect = [
            toplevel,
            OSError("git command failed"),
        ]
        assert get_git_ignored_files() == []


def test_get_git_ignored_files_no_ignored_files():
    """Test when git reports no ignored files."""
    with patch("subprocess.check_output") as mock_check_output:
        # rev-parse returns empty/toplevel, git ls-files returns empty
        mock_check_output.side_effect = ["", ""]

        files = get_git_ignored_files()
        assert files == []
        assert mock_check_output.call_count == 2


def test_get_git_ignored_files_mocked_success():
    """Test detecting ignored files with mocked git output."""
    toplevel = os.getcwd()
    with patch("subprocess.check_output") as mock_check_output, \
         patch("os.path.exists", return_value=True):

        mock_check_output.side_effect = [
            toplevel,
            "ignored1.py\nignored2.py\n"
        ]

        files = get_git_ignored_files()
        assert len(files) == 2
        expected = [os.path.join(toplevel, "ignored1.py"), os.path.join(toplevel, "ignored2.py")]
        assert files == expected


def test_get_git_ignored_files_real_git(tmp_path):
    """Verify that ignored files are correctly detected in a real git repository."""
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(tmp_path), check=True)

    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("ignored.py\n")

    tracked_file = tmp_path / "tracked.py"
    tracked_file.write_text("print('tracked')")
    subprocess.run(["git", "add", ".gitignore", "tracked.py"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True)

    ignored_file = tmp_path / "ignored.py"
    ignored_file.write_text("print('ignored')")

    untracked_file = tmp_path / "untracked.py"
    untracked_file.write_text("print('untracked')")

    results = get_git_ignored_files(str(tmp_path))
    result_paths = [os.path.abspath(r) for r in results]

    assert os.path.abspath(ignored_file) in result_paths
    assert os.path.abspath(untracked_file) not in result_paths
    assert os.path.abspath(tracked_file) not in result_paths


def test_scan_git_ignored_click(mocker):
    """Test that scan_git_ignored_click calls _generic_scan_click properly."""
    mock_generic = mocker.patch("gptscan._generic_scan_click")
    scan_git_ignored_click()
    assert mock_generic.call_count == 1
    args, kwargs = mock_generic.call_args
    assert args[1] == "Git Ignored Files"
    assert args[2] == "No ignored Git files detected in target folder."
    assert args[3] == "Git Ignored Files Error"


def test_cli_git_ignored_flag(tmp_path, mocker):
    """Test running CLI with --git-ignored."""
    ignored = tmp_path / "ignored.py"
    ignored.write_text("print('test')")

    scan_called_targets = []

    def mock_run_cli(scan_targets, *args, **kwargs):
        scan_called_targets.extend(scan_targets)
        return 0

    mocker.patch("gptscan.run_cli", mock_run_cli)
    mocker.patch("gptscan.get_git_ignored_files", return_value=[str(ignored)])

    with patch("sys.argv", ["gptscan.py", str(tmp_path), "--git-ignored", "--cli"]):
        gptscan.main()

    assert scan_called_targets == [str(ignored)]


def test_cli_git_ignored_auto_cli(tmp_path, mocker):
    """Test that --git-ignored triggers CLI mode without explicit --cli flag."""
    ignored = tmp_path / "ignored.py"
    ignored.write_text("print('test')")

    scan_called = []

    def mock_run_cli(scan_targets, *args, **kwargs):
        scan_called.extend(scan_targets)
        return 0

    mocker.patch("gptscan.run_cli", mock_run_cli)
    mocker.patch("gptscan.get_git_ignored_files", return_value=[str(ignored)])

    with patch("sys.argv", ["gptscan.py", "--git-ignored"]):
        gptscan.main()

    assert scan_called == [str(ignored)]
