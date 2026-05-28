import os
import pytest
from gptscan import _normalize_and_filter_dirs

def test_normalize_and_filter_dirs_basic(tmp_path):
    d1 = tmp_path / "dir1"
    d1.mkdir()
    d2 = tmp_path / "dir2"
    d2.mkdir()

    paths = [str(d1), str(d2)]
    result = _normalize_and_filter_dirs(paths)
    assert len(result) == 2
    assert os.path.abspath(str(d1)) in result
    assert os.path.abspath(str(d2)) in result

def test_normalize_and_filter_dirs_none_empty():
    paths = [None, ""]
    result = _normalize_and_filter_dirs(paths)
    assert result == []

def test_normalize_and_filter_dirs_deduplication(tmp_path):
    d1 = tmp_path / "dir1"
    d1.mkdir()

    abs_d1 = os.path.abspath(str(d1))

    # Using relative path to the same directory
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        rel_d1 = "dir1"
        paths = [abs_d1, rel_d1, abs_d1]
        result = _normalize_and_filter_dirs(paths)
        assert result == [abs_d1]
    finally:
        os.chdir(cwd)

def test_normalize_and_filter_dirs_non_existent(tmp_path):
    d1 = tmp_path / "non_existent"

    result = _normalize_and_filter_dirs([str(d1)])
    assert result == []

def test_normalize_and_filter_dirs_files(tmp_path):
    f1 = tmp_path / "file.txt"
    f1.write_text("hello")

    result = _normalize_and_filter_dirs([str(f1)])
    assert result == []

def test_normalize_and_filter_dirs_order(tmp_path):
    d1 = tmp_path / "dir1"
    d1.mkdir()
    d2 = tmp_path / "dir2"
    d2.mkdir()

    abs_d1 = os.path.abspath(str(d1))
    abs_d2 = os.path.abspath(str(d2))

    paths = [str(d2), str(d1), str(d2)]
    result = _normalize_and_filter_dirs(paths)
    assert result == [abs_d2, abs_d1]
