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
