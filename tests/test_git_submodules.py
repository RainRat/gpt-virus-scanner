import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import gptscan


def test_get_git_submodule_paths_not_git_repo(monkeypatch):
    """Verify that get_git_submodule_paths returns empty list when path is not in a git repo."""
    monkeypatch.setattr(gptscan, "_get_git_info", lambda p: (None, None))
    assert gptscan.get_git_submodule_paths("/some/path") == []


def test_get_git_submodule_paths_success(monkeypatch, tmp_path):
    """Verify get_git_submodule_paths extracts submodule paths from git submodule status output."""
    sub_dir = tmp_path / "libs" / "submod1"
    sub_dir.mkdir(parents=True)

    monkeypatch.setattr(gptscan, "_get_git_info", lambda p: (str(tmp_path), "."))

    mock_status_output = " e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 libs/submod1 (heads/main)\n"

    def mock_check_output(cmd, cwd=None, **kwargs):
        if "submodule" in cmd:
            return mock_status_output
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    res = gptscan.get_git_submodule_paths(str(tmp_path))
    assert len(res) == 1
    assert res[0] == str(sub_dir)


def test_get_git_submodule_paths_fallback_gitmodules(monkeypatch, tmp_path):
    """Verify fallback to parsing .gitmodules when git command fails."""
    sub_dir = tmp_path / "external" / "pkg"
    sub_dir.mkdir(parents=True)

    gitmodules = tmp_path / ".gitmodules"
    gitmodules.write_text(
        "[submodule \"pkg\"]\n"
        "\tpath = external/pkg\n"
        "\turl = https://github.com/example/pkg.git\n",
        encoding="utf-8"
    )

    monkeypatch.setattr(gptscan, "_get_git_info", lambda p: (str(tmp_path), "."))

    def mock_check_output_fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "check_output", mock_check_output_fail)

    res = gptscan.get_git_submodule_paths(str(tmp_path))
    assert len(res) == 1
    assert res[0] == str(sub_dir)


def test_get_git_submodule_paths_empty(monkeypatch, tmp_path):
    """Verify empty list returned when no submodules exist."""
    monkeypatch.setattr(gptscan, "_get_git_info", lambda p: (str(tmp_path), "."))
    monkeypatch.setattr(subprocess, "check_output", lambda cmd, **k: "")

    res = gptscan.get_git_submodule_paths(str(tmp_path))
    assert res == []


def test_scan_git_submodules_click(monkeypatch, tmp_path):
    """Verify scan_git_submodules_click calls _generic_scan_click properly."""
    sub_dir = tmp_path / "sub1"
    sub_dir.mkdir()

    called = []
    def mock_generic(func, title, failure_msg, error_title, is_snippets=False):
        called.append((title, func()))

    monkeypatch.setattr(gptscan, "_generic_scan_click", mock_generic)
    monkeypatch.setattr(gptscan, "get_git_submodule_paths", lambda p: [str(sub_dir)])
    monkeypatch.setattr(gptscan, "_get_target_path", lambda: str(tmp_path))

    gptscan.scan_git_submodules_click()

    assert len(called) == 1
    assert called[0][0] == "Git Submodules"
    assert called[0][1] == [str(sub_dir)]


def test_cli_git_submodules_option(monkeypatch, tmp_path):
    """Verify CLI --git-submodules collects submodule paths."""
    sub1 = tmp_path / "submodule1"
    sub1.mkdir()
    target_file = sub1 / "test_script.py"
    target_file.write_text("print('submodule')", encoding="utf-8")

    monkeypatch.setattr(gptscan, "get_git_submodule_paths", lambda p: [str(sub1)])

    scan_called_targets = []
    def mock_run_cli(scan_targets, *args, **kwargs):
        scan_called_targets.extend(scan_targets)
        return 0

    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)
    test_args = ["gptscan.py", str(tmp_path), "--git-submodules", "--cli"]
    monkeypatch.setattr("sys.argv", test_args)

    gptscan.main()

    assert str(sub1) in scan_called_targets
