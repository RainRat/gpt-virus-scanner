import os
from pathlib import Path
import pytest
from gptscan import collect_files

def test_collect_files_basic_glob(tmp_path, monkeypatch):
    """Test basic glob expansion like *.py."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").touch()
    (tmp_path / "b.js").touch()
    (tmp_path / "c.py").touch()

    targets = ["*.py"]
    results = collect_files(targets)

    paths = {p.name for p in results}
    assert paths == {"a.py", "c.py"}

def test_collect_files_recursive_glob(tmp_path, monkeypatch):
    """Test recursive glob expansion like **/*.js."""
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.js").touch()
    (src / "b.py").touch()

    nested = src / "nested"
    nested.mkdir()
    (nested / "c.js").touch()

    targets = ["**/*.js"]
    results = collect_files(targets)

    paths = {p.name for p in results}
    assert paths == {"a.js", "c.js"}

def test_collect_files_mixed_targets(tmp_path, monkeypatch):
    """Test a mix of literal paths and glob patterns."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").touch()
    (tmp_path / "b.js").touch()

    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.py").touch()

    # Literal file, Literal dir, and Glob
    targets = ["a.py", "sub", "*.js"]
    results = collect_files(targets)

    paths = {p.name for p in results}
    assert paths == {"a.py", "c.py", "b.js"}

def test_collect_files_glob_matching_dir(tmp_path, monkeypatch):
    """Test glob patterns that match directories."""
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").touch()

    other = tmp_path / "other"
    other.mkdir()
    (other / "b.js").touch()

    # Glob matching directories
    targets = ["sr*"]
    results = collect_files(targets)

    paths = {p.name for p in results}
    assert paths == {"a.py"}

def test_collect_files_no_match(tmp_path, monkeypatch):
    """Test glob patterns that match nothing."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").touch()

    targets = ["*.js", "non_existent.py"]
    results = collect_files(targets)

    assert results == []

def test_collect_files_literal_with_glob_chars(tmp_path, monkeypatch):
    """Test that literal paths containing glob chars are handled if they exist."""
    monkeypatch.chdir(tmp_path)
    # File named literally "file[1].py"
    special = tmp_path / "file[1].py"
    special.touch()

    # If it exists literally, it should be picked up
    targets = ["file[1].py"]
    results = collect_files(targets)

    paths = {p.name for p in results}
    assert paths == {"file[1].py"}

def test_collect_files_space_separated_string(tmp_path, monkeypatch):
    """Test that collect_files handles a single space-separated string of targets."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").touch()
    (tmp_path / "b.js").touch()

    targets = "a.py b.js"
    results = collect_files(targets)

    paths = {p.name for p in results}
    assert paths == {"a.py", "b.js"}
