from pathlib import Path
from unittest.mock import MagicMock, PropertyMock
import gptscan

def test_collect_files_single_file(tmp_path):
    f = tmp_path / "file.txt"
    f.touch()
    results = gptscan.collect_files(str(f))
    assert results == [f]

def test_collect_files_single_directory_recursive(tmp_path):
    d = tmp_path / "subdir"
    d.mkdir()
    f1 = d / "f1.txt"
    f1.touch()
    f2 = tmp_path / "f2.txt"
    f2.touch()

    # scan tmp_path
    results = gptscan.collect_files(str(tmp_path))
    assert set(results) == {f1, f2}

def test_collect_files_ignores_directories(tmp_path):
    """Ensure that only files are returned, and directories are ignored."""
    d = tmp_path / "subdir"
    d.mkdir()
    f = d / "file.txt"
    f.touch()
    empty_d = d / "empty_dir"
    empty_d.mkdir()

    results = gptscan.collect_files(str(tmp_path))
    assert set(results) == {f}
    assert empty_d not in results
    assert d not in results

def test_collect_files_mixed_input(tmp_path):
    f1 = tmp_path / "f1.txt"
    f1.touch()
    d = tmp_path / "dir"
    d.mkdir()
    f2 = d / "f2.txt"
    f2.touch()

    results = gptscan.collect_files([str(f1), str(d)])
    assert set(results) == {f1, f2}

def test_collect_files_deduplication(tmp_path):
    f1 = tmp_path / "f1.txt"
    f1.touch()

    results = gptscan.collect_files([str(f1), str(f1)])
    assert results == [f1]

def test_collect_files_deduplication_overlap(tmp_path):
    d = tmp_path / "dir"
    d.mkdir()
    f1 = d / "f1.txt"
    f1.touch()

    # Pass directory and the file inside it explicitly
    results = gptscan.collect_files([str(d), str(f1)])
    # Since dict.fromkeys preserves insertion order, and directory comes first,
    # the file from directory scan is added first.
    # If list_files returns [f1], then results becomes [f1, f1] -> [f1]
    assert results == [f1]

def test_collect_files_non_existent(tmp_path):
    results = gptscan.collect_files([str(tmp_path / "fake.txt")])
    assert results == []

def test_collect_files_string_input(tmp_path):
    f = tmp_path / "file.txt"
    f.touch()
    results = gptscan.collect_files(str(f))
    assert results == [f]

def test_collect_files_directory_as_list_element(tmp_path):
    d = tmp_path / "dir"
    d.mkdir()
    f = d / "file.txt"
    f.touch()
    results = gptscan.collect_files([str(d)])
    assert results == [f]


def test_collect_files_modified_since_stat_error(tmp_path, monkeypatch):
    f1 = tmp_path / "f1.txt"
    f2 = tmp_path / "f2.txt"
    f1.touch()
    f2.touch()

    original_stat = Path.stat

    def mock_stat(self, *args, **kwargs):
        st = original_stat(self, *args, **kwargs)
        if self.name == "f1.txt":
            mock_st = MagicMock()
            mock_st.st_mode = st.st_mode
            type(mock_st).st_mtime = PropertyMock(side_effect=OSError("Access denied"))
            return mock_st
        elif self.name == "f2.txt":
            mock_st = MagicMock()
            mock_st.st_mode = st.st_mode
            type(mock_st).st_mtime = PropertyMock(side_effect=FileNotFoundError("Missing file"))
            return mock_st
        return st

    monkeypatch.setattr(Path, "stat", mock_stat)

    results = gptscan.collect_files([str(f1), str(f2)], modified_since=100.0)
    assert results == []
