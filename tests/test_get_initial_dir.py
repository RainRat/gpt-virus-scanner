import os
import shlex
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from gptscan import _get_initial_dir, Config

@pytest.fixture
def mock_textbox(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("gptscan.textbox", mock)
    return mock

def test_get_initial_dir_textbox_directory(mock_textbox, tmp_path):
    d = tmp_path / "test_dir"
    d.mkdir()
    mock_textbox.get.return_value = str(d)

    assert _get_initial_dir() == str(d.absolute())

def test_get_initial_dir_textbox_file(mock_textbox, tmp_path):
    d = tmp_path / "test_dir"
    d.mkdir()
    f = d / "test_file.txt"
    f.write_text("hello")
    mock_textbox.get.return_value = str(f)

    # Should return the parent directory
    assert _get_initial_dir() == str(d.absolute())

def test_get_initial_dir_fallback_last_path(mock_textbox, tmp_path, monkeypatch):
    mock_textbox.get.return_value = ""
    d = tmp_path / "last_dir"
    d.mkdir()
    monkeypatch.setattr(Config, "last_path", str(d))

    assert _get_initial_dir() == str(d.absolute())

def test_get_initial_dir_fallback_last_path_file(mock_textbox, tmp_path, monkeypatch):
    mock_textbox.get.return_value = ""
    d = tmp_path / "last_dir"
    d.mkdir()
    f = d / "last_file.txt"
    f.write_text("hello")
    monkeypatch.setattr(Config, "last_path", str(f))

    assert _get_initial_dir() == str(d.absolute())

def test_get_initial_dir_empty(mock_textbox, monkeypatch):
    mock_textbox.get.return_value = ""
    monkeypatch.setattr(Config, "last_path", "")

    assert _get_initial_dir() is None

def test_get_initial_dir_multiple_paths(mock_textbox, tmp_path):
    d1 = tmp_path / "dir1"
    d1.mkdir()
    d2 = tmp_path / "dir2"
    d2.mkdir()

    # Simulate multiple paths in the textbox
    # Use shlex style quoting for paths with spaces if needed, but tmp_path usually doesn't have spaces.
    # However, let's just use shlex.quote to be safe.
    path1 = str(d1)
    path2 = str(d2)
    mock_textbox.get.return_value = f'{shlex.quote(path1)} {shlex.quote(path2)}'

    assert _get_initial_dir() == str(d1.absolute())

def test_get_initial_dir_url(mock_textbox):
    mock_textbox.get.return_value = "https://example.com"
    assert _get_initial_dir() is None

def test_get_initial_dir_virtual_path(mock_textbox):
    mock_textbox.get.return_value = "[Stdin]"
    assert _get_initial_dir() is None

def test_get_initial_dir_non_existent(mock_textbox, tmp_path):
    mock_textbox.get.return_value = str(tmp_path / "ghost")
    assert _get_initial_dir() is None

def test_get_initial_dir_exception(mock_textbox, monkeypatch):
    mock_textbox.get.return_value = "unclosed 'quote"
    # shlex.split should raise an exception here, which _get_initial_dir catches
    assert _get_initial_dir() is None

def test_get_initial_dir_no_textbox(monkeypatch, tmp_path):
    monkeypatch.setattr("gptscan.textbox", None)
    d = tmp_path / "last_dir"
    d.mkdir()
    monkeypatch.setattr(Config, "last_path", str(d))

    assert _get_initial_dir() == str(d.absolute())
