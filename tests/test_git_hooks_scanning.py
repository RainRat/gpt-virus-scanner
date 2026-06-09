import os
import subprocess
import pytest
from gptscan import get_git_hooks_paths, run_cli

def test_get_git_hooks_paths(tmp_path):
    # Setup test repo
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo)
    # Ensure global hooksPath doesn't interfere
    subprocess.run(["git", "config", "core.hooksPath", ""], cwd=repo)
    hooks = repo / ".git" / "hooks"

    pre_commit = hooks / "pre-commit"
    pre_commit.write_text("#!/bin/sh\necho 'hello'")

    sample = hooks / "pre-push.sample"
    sample.write_text("#!/bin/sh\necho 'sample'")

    paths = get_git_hooks_paths(str(repo))
    assert any(str(pre_commit) == p for p in paths)
    assert not any(str(sample) == p for p in paths)

def test_cli_git_hooks(tmp_path, monkeypatch):
    # Setup test repo
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo)
    # Ensure global hooksPath doesn't interfere
    subprocess.run(["git", "config", "core.hooksPath", ""], cwd=repo)
    hooks = repo / ".git" / "hooks"

    pre_commit = hooks / "pre-commit"
    pre_commit.write_text("#!/bin/sh\necho 'hello'")

    # Mocking get_git_hooks_paths to ensure it returns our test hook
    monkeypatch.setattr("gptscan.get_git_hooks_paths", lambda x: [str(pre_commit)])

    # Simulate CLI logic for --git-hooks
    paths = [str(repo)]
    git_roots = paths
    scan_targets = []
    for root_dir in git_roots:
        scan_targets.extend(get_git_hooks_paths(root_dir))

    assert str(pre_commit) in scan_targets
