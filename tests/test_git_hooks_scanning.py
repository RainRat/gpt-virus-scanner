import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from gptscan import get_git_hooks_paths

def test_get_git_hooks_paths_default_location():
    with patch("gptscan._get_git_info") as mock_info, \
         patch("subprocess.run") as mock_run, \
         patch("subprocess.check_output") as mock_check, \
         patch("os.path.isdir") as mock_isdir, \
         patch("os.listdir") as mock_listdir, \
         patch("os.path.isfile") as mock_isfile:

        mock_info.return_value = ("/repo", "rel")

        # git config --get core.hooksPath -> not set
        mock_run.return_value = MagicMock(stdout="", returncode=1)

        # git rev-parse --git-dir -> .git
        mock_check.return_value = ".git"

        mock_isdir.return_value = True
        mock_listdir.return_value = ["pre-commit", "pre-push.sample", "post-checkout"]
        mock_isfile.side_effect = lambda p: not p.endswith(".sample")

        paths = get_git_hooks_paths("/repo")

        assert len(paths) == 2
        assert any(p.endswith("pre-commit") for p in paths)
        assert any(p.endswith("post-checkout") for p in paths)
        assert not any(p.endswith(".sample") for p in paths)

def test_get_git_hooks_paths_custom_absolute_path():
    with patch("gptscan._get_git_info") as mock_info, \
         patch("subprocess.run") as mock_run, \
         patch("os.path.isdir") as mock_isdir, \
         patch("os.listdir") as mock_listdir, \
         patch("os.path.isfile") as mock_isfile:

        mock_info.return_value = ("/repo", ".")

        # git config --get core.hooksPath -> /custom/hooks
        mock_run.return_value = MagicMock(stdout="/custom/hooks\n", returncode=0)

        mock_isdir.side_effect = lambda p: p == "/custom/hooks"
        mock_listdir.return_value = ["hook1"]
        mock_isfile.return_value = True

        paths = get_git_hooks_paths("/repo")

        assert paths == ["/custom/hooks/hook1"]

def test_get_git_hooks_paths_custom_relative_path():
    with patch("gptscan._get_git_info") as mock_info, \
         patch("subprocess.run") as mock_run, \
         patch("os.path.isdir") as mock_isdir, \
         patch("os.listdir") as mock_listdir, \
         patch("os.path.isfile") as mock_isfile:

        repo_root = os.path.abspath("/repo")
        mock_info.return_value = (repo_root, ".")

        # git config --get core.hooksPath -> myhooks
        mock_run.return_value = MagicMock(stdout="myhooks\n", returncode=0)

        expected_dir = os.path.join(repo_root, "myhooks")
        mock_isdir.side_effect = lambda p: p == expected_dir
        mock_listdir.return_value = ["hook2"]
        mock_isfile.return_value = True

        paths = get_git_hooks_paths("/repo")

        assert paths == [os.path.join(expected_dir, "hook2")]

def test_get_git_hooks_paths_tilde_expansion():
    with patch("gptscan._get_git_info") as mock_info, \
         patch("subprocess.run") as mock_run, \
         patch("pathlib.Path.expanduser") as mock_expand, \
         patch("os.path.isdir") as mock_isdir, \
         patch("os.listdir") as mock_listdir, \
         patch("os.path.isfile") as mock_isfile:

        mock_info.return_value = ("/repo", ".")
        mock_run.return_value = MagicMock(stdout="~/hooks\n", returncode=0)

        expanded_path = Path("/home/user/hooks")
        mock_expand.return_value = expanded_path

        mock_isdir.side_effect = lambda p: str(p) == str(expanded_path)
        mock_listdir.return_value = ["hook3"]
        mock_isfile.return_value = True

        paths = get_git_hooks_paths("/repo")

        assert paths == [str(expanded_path / "hook3")]

def test_get_git_hooks_paths_no_repo():
    with patch("gptscan._get_git_info") as mock_info:
        mock_info.return_value = (None, None)

        # When not in a repo and no global hooksPath set, should return empty
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=1)
            paths = get_git_hooks_paths("/not/a/repo")
            assert paths == []

def test_get_git_hooks_paths_exception_fallback():
    with patch("gptscan._get_git_info") as mock_info, \
         patch("subprocess.run") as mock_run, \
         patch("os.path.isdir") as mock_isdir, \
         patch("os.listdir") as mock_listdir, \
         patch("os.path.isfile") as mock_isfile:

        mock_info.return_value = ("/repo", ".")

        # git config fails or raises
        mock_run.side_effect = OSError("git not found")

        expected_dir = os.path.join("/repo", ".git", "hooks")
        mock_isdir.side_effect = lambda p: p == expected_dir
        mock_listdir.return_value = ["fallback-hook"]
        mock_isfile.return_value = True

        paths = get_git_hooks_paths("/repo")

        assert paths == [os.path.join(expected_dir, "fallback-hook")]
