import os
import subprocess
from unittest.mock import patch, MagicMock
import pytest
from gptscan import get_git_changed_files, get_git_diff

def test_get_git_changed_files_with_revision():
    """Test that the revision argument is correctly passed to git diff."""
    with patch("subprocess.check_output") as mock_check_output, \
         patch("os.path.exists") as mock_exists:

        # 1. rev-parse (to get toplevel)
        # 2. git diff --name-only revision -- target
        mock_check_output.side_effect = ["/repo", "file1.py\nfile2.py\n"]
        mock_exists.return_value = True

        revision = "main..feature"
        files = get_git_changed_files("/repo/subdir", ref=revision)

        # rev-parse
        assert "rev-parse" in mock_check_output.call_args_list[0][0][0]

        # git diff
        args, kwargs = mock_check_output.call_args_list[1]
        cmd = args[0]
        assert cmd[0] == "git"
        assert cmd[1] == "diff"
        assert cmd[2] == "--name-only"
        assert cmd[3] == revision

        # ls-files should NOT be called when ref != "HEAD"
        assert mock_check_output.call_count == 2
        assert len(files) == 2

def test_get_git_diff_with_revision():
    """Test that get_git_diff correctly uses the revision argument."""
    with patch("subprocess.check_output") as mock_check_output:
        # 1. rev-parse
        # 2. git diff revision
        mock_check_output.side_effect = ["/repo", "diff content"]

        revision = "HEAD~1"
        diff = get_git_diff("/repo", ref=revision)

        assert diff == "diff content"

        # git diff
        args, kwargs = mock_check_output.call_args_list[1]
        cmd = args[0]
        assert cmd[0] == "git"
        assert cmd[1] == "diff"
        assert cmd[2] == revision
        assert "--no-color" in cmd

def test_get_git_changed_files_default_head_includes_untracked():
    """Verify that untracked files are still included by default (HEAD)."""
    with patch("subprocess.check_output") as mock_check_output, \
         patch("os.path.exists") as mock_exists:

        # 1. rev-parse
        # 2. git diff
        # 3. git ls-files
        mock_check_output.side_effect = ["/repo", "changed.py", "untracked.py"]
        mock_exists.return_value = True

        files = get_git_changed_files("/repo")

        assert mock_check_output.call_count == 3
        assert len(files) == 2
        assert any("changed.py" in f for f in files)
        assert any("untracked.py" in f for f in files)

def test_get_git_changed_files_real_git_revision(tmp_path):
    """Real git test for revision scanning."""
    # Initialize repo
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "you@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.name", "Your Name"], cwd=str(tmp_path), check=True)

    # First commit
    f1 = tmp_path / "file1.py"
    f1.write_text("print('v1')")
    subprocess.run(["git", "add", "file1.py"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=str(tmp_path), check=True)

    # Second commit
    f2 = tmp_path / "file2.py"
    f2.write_text("print('v2')")
    subprocess.run(["git", "add", "file2.py"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "second"], cwd=str(tmp_path), check=True)

    # Scan difference between HEAD~1 and HEAD
    results = get_git_changed_files(str(tmp_path), ref="HEAD~1..HEAD")

    result_paths = [os.path.basename(r) for r in results]
    assert "file2.py" in result_paths
    assert "file1.py" not in result_paths
