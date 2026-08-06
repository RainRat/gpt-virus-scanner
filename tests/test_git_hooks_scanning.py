import os
import subprocess
from pathlib import Path
import pytest
from unittest.mock import MagicMock
import gptscan
from gptscan import get_git_hooks_paths, run_cli

def test_get_git_hooks_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo)
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
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo)
    subprocess.run(["git", "config", "core.hooksPath", ""], cwd=repo)
    hooks = repo / ".git" / "hooks"

    pre_commit = hooks / "pre-commit"
    pre_commit.write_text("#!/bin/sh\necho 'hello'")

    monkeypatch.setattr("gptscan.get_git_hooks_paths", lambda x: [str(pre_commit)])

    paths = [str(repo)]
    git_roots = paths
    scan_targets = []
    for root_dir in git_roots:
        scan_targets.extend(get_git_hooks_paths(root_dir))

    assert str(pre_commit) in scan_targets

def test_get_git_hooks_paths_with_absolute_core_hookspath(monkeypatch, tmp_path):
    hooks_dir = tmp_path / "global_hooks"
    hooks_dir.mkdir()
    pre_commit = hooks_dir / "pre-commit"
    pre_commit.write_text("#!/bin/sh\necho 'global'")
    sample = hooks_dir / "pre-commit.sample"
    sample.write_text("#!/bin/sh\necho 'sample'")

    mock_res = MagicMock()
    mock_res.stdout = str(hooks_dir) + "\n"
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: mock_res)
    monkeypatch.setattr("gptscan._get_git_info", lambda path: ("/repo/root", "some_file"))

    paths = get_git_hooks_paths("/repo/root")
    assert str(pre_commit) in paths
    assert str(sample) not in paths

def test_get_git_hooks_paths_with_relative_core_hookspath(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    hooks_dir = repo_root / "relative_hooks"
    hooks_dir.mkdir()
    pre_commit = hooks_dir / "pre-commit"
    pre_commit.write_text("#!/bin/sh\necho 'relative'")

    mock_res = MagicMock()
    mock_res.stdout = "relative_hooks\n"
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: mock_res)
    monkeypatch.setattr("gptscan._get_git_info", lambda path: (str(repo_root), "some_file"))

    paths = get_git_hooks_paths(str(repo_root))
    assert str(pre_commit) in paths

def test_get_git_hooks_paths_fallback_to_absolute_git_dir(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    git_dir = tmp_path / "custom_git_dir"
    git_dir.mkdir()
    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir()
    pre_commit = hooks_dir / "pre-commit"
    pre_commit.write_text("#!/bin/sh\necho 'fallback'")

    mock_res = MagicMock()
    mock_res.stdout = ""
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: mock_res)

    def mock_check_output(args, **kwargs):
        if args == ["git", "rev-parse", "--git-dir"]:
            return str(git_dir) + "\n"
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)
    monkeypatch.setattr("gptscan._get_git_info", lambda path: (str(repo_root), "some_file"))

    paths = get_git_hooks_paths(str(repo_root))
    assert str(pre_commit) in paths

def test_get_git_hooks_paths_fallback_to_relative_git_dir(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    git_dir = repo_root / "custom_relative_git_dir"
    git_dir.mkdir()
    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir()
    pre_commit = hooks_dir / "pre-commit"
    pre_commit.write_text("#!/bin/sh\necho 'relative fallback'")

    mock_res = MagicMock()
    mock_res.stdout = ""
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: mock_res)

    def mock_check_output(args, **kwargs):
        if args == ["git", "rev-parse", "--git-dir"]:
            return "custom_relative_git_dir\n"
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)
    monkeypatch.setattr("gptscan._get_git_info", lambda path: (str(repo_root), "some_file"))

    paths = get_git_hooks_paths(str(repo_root))
    assert str(pre_commit) in paths

def test_get_git_hooks_paths_subprocess_exceptions(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    hooks_dir = repo_root / ".git" / "hooks"
    hooks_dir.mkdir(parents=True)
    pre_commit = hooks_dir / "pre-commit"
    pre_commit.write_text("#!/bin/sh\necho 'exception fallback'")

    def mock_run_raise(*args, **kwargs):
        raise OSError("Subprocess failed")

    monkeypatch.setattr(subprocess, "run", mock_run_raise)
    monkeypatch.setattr("gptscan._get_git_info", lambda path: (str(repo_root), "some_file"))

    paths = get_git_hooks_paths(str(repo_root))
    assert str(pre_commit) in paths

def test_get_git_hooks_paths_listdir_os_error(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    mock_res = MagicMock()
    mock_res.stdout = "some_hooks\n"
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: mock_res)
    monkeypatch.setattr("gptscan._get_git_info", lambda path: (str(repo_root), "some_file"))
    monkeypatch.setattr(os.path, "isdir", lambda path: True)

    def mock_listdir_raise(path):
        raise OSError("Permission denied")

    monkeypatch.setattr(os, "listdir", mock_listdir_raise)

    paths = get_git_hooks_paths(str(repo_root))
    assert paths == []
