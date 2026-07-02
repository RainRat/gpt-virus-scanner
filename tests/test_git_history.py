import pytest
from unittest.mock import patch, MagicMock
import gptscan
import subprocess

def test_get_git_history_snippets_no_git(monkeypatch):
    monkeypatch.setattr("gptscan._get_git_info", lambda p: (None, None))
    snippets = gptscan.get_git_history_snippets(".", count=5)
    assert snippets == []

def test_get_git_history_snippets_success(monkeypatch):
    monkeypatch.setattr("gptscan._get_git_info", lambda p: ("/repo", "."))

    def mock_check_output(cmd, cwd=None, **kwargs):
        if cmd[1] == "rev-list":
            return "hash1\nhash2\n"
        elif cmd[1] == "show":
            return f"commit {cmd[3]}\nAuthor: test\n\ndiff --git a/file.py b/file.py\n+new line"
        return ""

    monkeypatch.setattr("subprocess.check_output", mock_check_output)

    snippets = gptscan.get_git_history_snippets(".", count=2)
    assert len(snippets) == 2
    assert snippets[0][0] == "[Git History] commit hash1"
    assert b"commit hash1" in snippets[0][1]
    assert snippets[1][0] == "[Git History] commit hash2"
    assert b"commit hash2" in snippets[1][1]

def test_cli_git_history_flag(monkeypatch):
    monkeypatch.setattr("gptscan.get_git_history_snippets", lambda p, count: [("commit1", b"diff1")])

    cli_extra_snippets = []
    def mock_run_cli(*args, **kwargs):
        nonlocal cli_extra_snippets
        cli_extra_snippets = kwargs.get("extra_snippets", [])
        return 0

    monkeypatch.setattr("gptscan.run_cli", mock_run_cli)

    import sys
    test_args = ["gptscan.py", "--git-history", "3", "--cli"]
    with patch.object(sys, 'argv', test_args):
        gptscan.main()

    assert len(cli_extra_snippets) == 1
    assert cli_extra_snippets[0] == ("commit1", b"diff1")

def test_unpack_git_commit(monkeypatch):
    content = b"commit hash123\nAuthor: test\n\n--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-old\n+new\n"
    snippets = list(gptscan.unpack_content("test.patch", content))
    assert len(snippets) == 1
    assert "test.patch [test.py @ line 1]" in snippets[0][0]
    assert b"new" in snippets[0][1]
