import pytest
from unittest.mock import MagicMock, patch
import gptscan
import subprocess

def test_get_git_history_snippets_partial_failure(monkeypatch):
    """Verify that a single commit failure doesn't cause the whole history scan to fail."""
    monkeypatch.setattr("gptscan._get_git_info", lambda p: ("/repo", "."))

    def mock_check_output(cmd, cwd=None, **kwargs):
        if "rev-list" in cmd:
            return "hash1\nhash2\nhash3\n"
        elif "show" in cmd:
            if "hash2" in cmd:
                # Simulate failure for the second hash
                raise subprocess.CalledProcessError(1, cmd)
            return f"commit {cmd[-1]}\nAuthor: test\n\ndiff --git a/file.py b/file.py\n+new line"
        return ""

    monkeypatch.setattr("subprocess.check_output", mock_check_output)

    # After fix: should return 2 snippets (hash1 and hash3)
    snippets = gptscan.get_git_history_snippets(".", count=3)
    assert len(snippets) == 2
    assert snippets[0][0] == "[Git History] commit hash1"
    assert snippets[1][0] == "[Git History] commit hash3"
