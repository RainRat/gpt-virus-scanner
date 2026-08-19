import os
import subprocess
import pytest
from pathlib import Path
from gptscan import get_git_stash_snippets

def test_get_git_stash_snippets(tmp_path):
    # Initialize a git repo
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "you@example.com"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Your Name"], cwd=repo_dir, check=True)

    # Create a file and commit it
    test_file = repo_dir / "test.py"
    test_file.write_text("print('hello')")
    subprocess.run(["git", "add", "test.py"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=repo_dir, check=True)

    # Modify the file and stash it
    test_file.write_text("print('hello world')")
    subprocess.run(["git", "stash"], cwd=repo_dir, check=True)

    # Get snippets
    snippets = get_git_stash_snippets(str(repo_dir))

    assert len(snippets) >= 1
    name, content = snippets[0]
    assert "stash@{0}" in name
    assert b"print('hello world')" in content

def test_get_git_stash_snippets_no_stash(tmp_path):
    # Initialize a git repo
    repo_dir = tmp_path / "repo_no_stash"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=repo_dir, check=True)

    # Get snippets
    snippets = get_git_stash_snippets(str(repo_dir))
    assert snippets == []

def test_get_git_stash_snippets_non_git(tmp_path):
    non_git_dir = tmp_path / "non_git"
    non_git_dir.mkdir()

    # Get snippets
    snippets = get_git_stash_snippets(str(non_git_dir))
    assert snippets == []


def test_get_git_stash_snippets_empty_and_unmatched_lines(monkeypatch):
    monkeypatch.setattr("gptscan._get_git_info", lambda p: ("/mock/repo", ".git"))

    def mock_check_output(cmd, cwd=None, stderr=None, universal_newlines=None):
        if cmd == ["git", "stash", "list"]:
            return "\n   \nInvalid line\nstash@{0}: WIP on main\nstash@{1}: Empty diff\n"
        elif cmd == ["git", "stash", "show", "-p", "stash@{0}"]:
            return "diff --git a/f.py b/f.py\n+print('hello')"
        elif cmd == ["git", "stash", "show", "-p", "stash@{1}"]:
            return "   \n"
        return ""

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    snippets = get_git_stash_snippets("/mock/repo")
    assert len(snippets) == 1
    assert snippets[0][0] == "[stash@{0}] stash@{0}: WIP on main"
    assert b"print('hello')" in snippets[0][1]


def test_get_git_stash_snippets_show_process_error(monkeypatch):
    monkeypatch.setattr("gptscan._get_git_info", lambda p: ("/mock/repo", ".git"))

    def mock_check_output(cmd, cwd=None, stderr=None, universal_newlines=None):
        if cmd == ["git", "stash", "list"]:
            return "stash@{0}: Error stash\nstash@{1}: Valid stash\n"
        elif cmd == ["git", "stash", "show", "-p", "stash@{0}"]:
            raise subprocess.CalledProcessError(1, cmd)
        elif cmd == ["git", "stash", "show", "-p", "stash@{1}"]:
            return "+valid content"
        return ""

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    snippets = get_git_stash_snippets("/mock/repo")
    assert len(snippets) == 1
    assert snippets[0][0] == "[stash@{1}] stash@{1}: Valid stash"


def test_get_git_stash_snippets_command_exception(monkeypatch):
    monkeypatch.setattr("gptscan._get_git_info", lambda p: ("/mock/repo", ".git"))

    def mock_check_output_raise(cmd, cwd=None, stderr=None, universal_newlines=None):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "check_output", mock_check_output_raise)

    snippets = get_git_stash_snippets("/mock/repo")
    assert snippets == []


def test_cli_git_stash_flag(monkeypatch, tmp_path):
    import gptscan

    stashes = [("[stash@{0}] stash@{0}: WIP on main", b"eval(input())")]
    monkeypatch.setattr("gptscan.get_git_stash_snippets", lambda path: stashes)

    scanned_extra = []

    def mock_scan_files(*args, **kwargs):
        extra = kwargs.get("extra_snippets")
        if extra:
            scanned_extra.extend(extra)
        return [("result", ("stash@{0}", "100%", "Admin", "User", "90%", "eval(input())", "1"))]

    monkeypatch.setattr("gptscan.scan_files", mock_scan_files)

    # Invoke main CLI entry point with --git-stash argument
    test_args = ["gptscan.py", "--cli", "--git-stash", str(tmp_path)]
    monkeypatch.setattr("sys.argv", test_args)

    gptscan.main()

    assert scanned_extra == stashes
