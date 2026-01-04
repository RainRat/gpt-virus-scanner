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
        mock_check_output.assert_called_once()

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
        assert set(files) == {"staged.py", "unstaged.py", "untracked.py"}
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

        assert files == ["existing.py"]
        assert "deleted.py" not in files

def test_path_argument_passed():
    """Test that the path argument is correctly passed to git command cwd."""
    with patch("subprocess.check_output") as mock_check_output:
        mock_check_output.side_effect = ["", ""]

        target_dir = "/some/path"
        get_git_changed_files(target_dir)

        # Verify cwd was set correctly in both calls
        args1, kwargs1 = mock_check_output.call_args_list[0]
        args2, kwargs2 = mock_check_output.call_args_list[1]

        assert kwargs1["cwd"] == target_dir
        assert kwargs2["cwd"] == target_dir
