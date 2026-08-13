import os
import subprocess
import pytest
import gptscan

def test_get_git_info_directory_in_real_git_repository(tmp_path):
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    toplevel, rel_target = gptscan._get_git_info(str(subdir))

    assert toplevel is not None
    assert os.path.abspath(toplevel) == os.path.abspath(tmp_path)
    assert rel_target == "subdir"

def test_get_git_info_file_in_real_git_repository(tmp_path):
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    test_file = tmp_path / "test.txt"
    test_file.touch()

    toplevel, rel_target = gptscan._get_git_info(str(test_file))

    assert toplevel is not None
    assert os.path.abspath(toplevel) == os.path.abspath(tmp_path)
    assert rel_target == "test.txt"

def test_get_git_info_subprocess_called_process_error(monkeypatch):
    def mock_check_output(*args, **kwargs):
        raise subprocess.CalledProcessError(128, ["git"])
    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    toplevel, rel_target = gptscan._get_git_info(".")

    assert toplevel is None
    assert rel_target is None

def test_get_git_info_subprocess_file_not_found_error(monkeypatch):
    def mock_check_output(*args, **kwargs):
        raise FileNotFoundError("git not found")
    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    toplevel, rel_target = gptscan._get_git_info(".")

    assert toplevel is None
    assert rel_target is None

def test_get_git_info_subprocess_os_error(monkeypatch):
    def mock_check_output(*args, **kwargs):
        raise OSError("Permission denied")
    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    toplevel, rel_target = gptscan._get_git_info(".")

    assert toplevel is None
    assert rel_target is None

def test_get_git_info_os_relpath_value_error(monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "/some/git/repo\n")

    def mock_relpath(*args, **kwargs):
        raise ValueError("Paths on different drives")
    monkeypatch.setattr(os.path, "relpath", mock_relpath)

    toplevel, rel_target = gptscan._get_git_info(".")

    assert toplevel is None
    assert rel_target is None
