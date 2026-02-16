import os
import subprocess
from unittest.mock import patch, MagicMock
import pytest
from gptscan import get_git_changed_files

def test_git_not_found_or_error():
    """Test behavior when git command fails (e.g. not a repo or git not installed)."""
    with patch("subprocess.check_output") as mock_check_output:
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "git")

        files = get_git_changed_files()

        # Should return empty list and handle exception gracefully
        assert files == []
        # Both diff and ls-files are attempted
        assert mock_check_output.call_count == 2

def test_no_changes():
    """Test when git reports no changed files."""
    with patch("subprocess.check_output") as mock_check_output:
        # Both git diff and git ls-files return empty output
        mock_check_output.side_effect = ["", ""]

        files = get_git_changed_files()

        assert files == []
        assert mock_check_output.call_count == 2

def test_changes_detected():
    """Test detecting mixed staged, unstaged, and untracked files."""
    with patch("subprocess.check_output") as mock_check_output, \
         patch("os.path.exists") as mock_exists:

        # First call: git diff returns "staged.py\nunstaged.py"
        # Second call: git ls-files returns "untracked.py\nstaged.py" (overlap)
        mock_check_output.side_effect = [
            "staged.py\nunstaged.py\n",
            "untracked.py\nstaged.py\n"
        ]

        # All files exist
        mock_exists.return_value = True

        files = get_git_changed_files()

        # Result should be unique list of all 3 files
        assert len(files) == 3
        assert set(files) == {os.path.join(".", "staged.py"), os.path.join(".", "unstaged.py"), os.path.join(".", "untracked.py")}
        assert mock_check_output.call_count == 2

def test_file_does_not_exist():
    """Test that files deleted from disk are excluded even if git lists them."""
    with patch("subprocess.check_output") as mock_check_output, \
         patch("os.path.exists") as mock_exists:

        mock_check_output.side_effect = [
            "existing.py\ndeleted.py\n",
            ""
        ]

        def exists_side_effect(path):
            return "deleted.py" not in path

        mock_exists.side_effect = exists_side_effect

        files = get_git_changed_files()

        assert files == [os.path.join(".", "existing.py")]
        assert os.path.join(".", "deleted.py") not in files

def test_path_argument_passed():
    """Test that the path argument is correctly passed to git command cwd."""
    with patch("subprocess.check_output") as mock_check_output, \
         patch("os.path.exists") as mock_exists:
        mock_check_output.side_effect = ["changed.py", ""]
        mock_exists.return_value = True

        target_dir = "/some/path"
        files = get_git_changed_files(target_dir)

        assert files == [os.path.join(target_dir, "changed.py")]

        # Verify cwd was set correctly in both calls
        args1, kwargs1 = mock_check_output.call_args_list[0]
        args2, kwargs2 = mock_check_output.call_args_list[1]

        assert kwargs1["cwd"] == target_dir
        assert kwargs2["cwd"] == target_dir

def test_git_commands_use_relative_flag():
    """Verify that both git diff and git ls-files include the --relative flag."""
    with patch("subprocess.check_output") as mock_check_output, \
         patch("os.path.exists") as mock_exists:
        mock_check_output.side_effect = ["", ""]
        mock_exists.return_value = True

        get_git_changed_files("/some/path")

        # First call should be git diff
        args1, _ = mock_check_output.call_args_list[0]
        cmd1 = args1[0]
        assert "git" in cmd1
        assert "diff" in cmd1
        assert "--relative" in cmd1

        # Second call should be git ls-files
        args2, _ = mock_check_output.call_args_list[1]
        cmd2 = args2[0]
        assert "git" in cmd2
        assert "ls-files" in cmd2
        assert "--relative" in cmd2
