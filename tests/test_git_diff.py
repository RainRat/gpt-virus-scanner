import subprocess
import pytest
from pathlib import Path
from gptscan import get_git_diff

def test_get_git_diff_no_repo(tmp_path, monkeypatch):
    """Test get_git_diff in a directory that is not a git repo."""
    monkeypatch.chdir(tmp_path)
    assert get_git_diff() == ""

def test_get_git_diff_with_changes(tmp_path, monkeypatch):
    """Test get_git_diff with staged and unstaged changes."""
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init"], check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], check=True)

    file1 = tmp_path / "file1.txt"
    file1.write_text("initial content")
    subprocess.run(["git", "add", "file1.txt"], check=True)
    subprocess.run(["git", "commit", "-m", "initial"], check=True)

    # Unstaged change
    file1.write_text("unstaged change")

    diff = get_git_diff()
    assert "unstaged change" in diff
    assert "file1.txt" in diff

    # Staged change
    file2 = tmp_path / "file2.txt"
    file2.write_text("staged content")
    subprocess.run(["git", "add", "file2.txt"], check=True)

    diff = get_git_diff()
    assert "staged content" in diff
    assert "file2.txt" in diff

def test_get_git_diff_specific_path(tmp_path, monkeypatch):
    """Test get_git_diff on a specific path."""
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init"], check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], check=True)

    subdir = tmp_path / "subdir"
    subdir.mkdir()
    file1 = subdir / "file1.txt"
    file1.write_text("initial content")
    subprocess.run(["git", "add", "subdir/file1.txt"], check=True)
    subprocess.run(["git", "commit", "-m", "initial"], check=True)

    file1.write_text("changed content")

    # Diff of the whole repo
    assert "subdir/file1.txt" in get_git_diff()

    # Diff of a different (empty) subdir
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    assert get_git_diff(str(other_dir)) == ""

    # Diff of the specific subdir
    assert "subdir/file1.txt" in get_git_diff(str(subdir))
