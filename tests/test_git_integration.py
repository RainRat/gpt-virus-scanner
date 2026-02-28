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
        # rev-parse is attempted first
        assert mock_check_output.call_count == 1

def test_no_changes():
    """Test when git reports no changed files."""
    with patch("subprocess.check_output") as mock_check_output:
        # git rev-parse, git diff, and git ls-files return empty/success output
        mock_check_output.side_effect = ["", "", ""]

        files = get_git_changed_files()

        assert files == []
        assert mock_check_output.call_count == 3

def test_changes_detected():
    """Test detecting mixed staged, unstaged, and untracked files."""
    with patch("subprocess.check_output") as mock_check_output, \
         patch("os.path.exists") as mock_exists:

        # First call: rev-parse
        # Second call: git diff returns "staged.py\nunstaged.py"
        # Third call: git ls-files returns "untracked.py\nstaged.py" (overlap)
        mock_check_output.side_effect = [
            "",
            "staged.py\nunstaged.py\n",
            "untracked.py\nstaged.py\n"
        ]

        # All files exist
        mock_exists.return_value = True

        files = get_git_changed_files()

        # Result should be unique list of all 3 files
        assert len(files) == 3
        # Use abspath because get_git_changed_files now uses abspath for cwd
        cwd = os.getcwd()
        expected = {os.path.join(cwd, "staged.py"), os.path.join(cwd, "unstaged.py"), os.path.join(cwd, "untracked.py")}
        assert set(os.path.abspath(f) for f in files) == expected
        assert mock_check_output.call_count == 3

def test_file_does_not_exist():
    """Test that files deleted from disk are excluded even if git lists them."""
    with patch("subprocess.check_output") as mock_check_output, \
         patch("os.path.exists") as mock_exists:

        mock_check_output.side_effect = [
            "",
            "existing.py\ndeleted.py\n",
            ""
        ]

        def exists_side_effect(path):
            return "deleted.py" not in path

        mock_exists.side_effect = exists_side_effect

        files = get_git_changed_files()

        cwd = os.getcwd()
        assert len(files) == 1
        assert os.path.abspath(files[0]) == os.path.join(cwd, "existing.py")

def test_path_argument_passed():
    """Test that the path argument is correctly passed to git command cwd."""
    with patch("subprocess.check_output") as mock_check_output, \
         patch("os.path.exists") as mock_exists:
        mock_check_output.side_effect = ["", "changed.py", ""]
        mock_exists.return_value = True

        target_dir = os.path.abspath("/some/path")
        # Ensure target_dir exists for get_git_changed_files' os.path.exists check on results
        with patch("os.path.exists", return_value=True):
            files = get_git_changed_files(target_dir)

        # Verify cwd was set correctly in all calls (rev-parse, diff, ls-files)
        for call in mock_check_output.call_args_list:
            assert call[1]["cwd"] == target_dir

def test_git_commands_use_relative_flag():
    """Verify that git commands are called correctly."""
    with patch("subprocess.check_output") as mock_check_output, \
         patch("os.path.exists") as mock_exists:
        mock_check_output.side_effect = ["", "", ""]
        mock_exists.return_value = True

        get_git_changed_files("/some/path")

        # rev-parse, diff, ls-files
        assert mock_check_output.call_count == 3
        
        # Second call should be git diff
        args1, _ = mock_check_output.call_args_list[1]
        cmd1 = args1[0]
        assert "git" in cmd1
        assert "diff" in cmd1
        assert "--" in cmd1

        # Third call should be git ls-files
        args2, _ = mock_check_output.call_args_list[2]
        cmd2 = args2[0]
        assert "git" in cmd2
        assert "ls-files" in cmd2
        assert "--" in cmd2

def test_get_git_changed_files_with_file_path(tmp_path):
    """Test that get_git_changed_files handles being passed a file path instead of a directory."""
    # Initialize a git repo
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    f = tmp_path / "test.txt"
    f.touch()

    # It should detect the untracked file when targeted
    results = get_git_changed_files(str(f))
    assert any(str(f) == os.path.abspath(r) for r in results)

def test_get_git_changed_files_untracked_real_git(tmp_path):
    """Verify that untracked files are correctly detected in a real git repository."""
    # Initialize a git repo
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    # Create an untracked file
    untracked = tmp_path / "untracked.py"
    untracked.touch()

    # Run get_git_changed_files
    results = get_git_changed_files(str(tmp_path))

    # It should find the untracked file.
    assert any(str(untracked) == os.path.abspath(r) for r in results)
