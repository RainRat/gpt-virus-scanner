import pytest
from unittest.mock import MagicMock, patch
import subprocess
import os
from gptscan import get_git_conflict_snippets

def test_get_git_conflict_snippets_no_git():
    """Test that it returns empty list if not in a Git repository."""
    with patch("gptscan._get_git_info", return_value=(None, None)):
        assert get_git_conflict_snippets() == []

def test_get_git_conflict_snippets_no_conflicts():
    """Test that it returns empty list if no conflicts are found."""
    with patch("gptscan._get_git_info", return_value=("/repo", ".")):
        with patch("subprocess.check_output") as mock_run:
            # Mock git diff --name-only --diff-filter=U returning nothing
            mock_run.return_value = ""
            assert get_git_conflict_snippets("/repo") == []
            mock_run.assert_called_with(
                ["git", "diff", "--name-only", "--diff-filter=U"],
                cwd="/repo",
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

def test_get_git_conflict_snippets_with_conflicts():
    """Test that it returns snippets for unmerged files."""
    with patch("gptscan._get_git_info", return_value=("/repo", ".")):
        with patch("subprocess.check_output") as mock_run:
            # First call: get unmerged files (universal_newlines=True)
            # Second and third calls: get diff for each file (raw bytes)
            mock_run.side_effect = [
                "file1.py\nfile2.js\n",
                b"diff content for file1",
                b"diff content for file2"
            ]

            snippets = get_git_conflict_snippets("/repo")

            assert len(snippets) == 2
            assert snippets[0] == ("[Git Conflict] file1.py", b"diff content for file1")
            assert snippets[1] == ("[Git Conflict] file2.js", b"diff content for file2")

def test_get_git_conflict_snippets_error():
    """Test that it handles subprocess errors gracefully."""
    with patch("gptscan._get_git_info", return_value=("/repo", ".")):
        with patch("subprocess.check_output") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")
            assert get_git_conflict_snippets("/repo") == []
