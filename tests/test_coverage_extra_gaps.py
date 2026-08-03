import os
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import gptscan

def test_get_item_raw_values_corrupted_json(monkeypatch):
    mock_tree = MagicMock()
    mock_tree.exists.return_value = True
    mock_tree.item.return_value = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "invalid json string"]
    monkeypatch.setattr("gptscan.tree", mock_tree)

    result = gptscan._get_item_raw_values("item_id")
    assert result == ["v1", "v2", "v3", "v4", "v5", "v6", "v7"]

def test_resolve_file_paths_warning_box(monkeypatch):
    mock_showwarning = MagicMock()
    monkeypatch.setattr("gptscan.messagebox.showwarning", mock_showwarning)

    non_existent_path = "/this/path/does/not/exist/at/all"
    result = gptscan._resolve_file_paths(non_existent_path, verify=True)

    assert result == []
    mock_showwarning.assert_called_once_with(
        "Files Not Found", "The selected file(s) could not be located on disk."
    )

def test_get_php_packages_paths_exception(monkeypatch):
    def mock_check_output(*args, **kwargs):
        raise RuntimeError("Composer failure")
    monkeypatch.setattr("subprocess.check_output", mock_check_output)

    monkeypatch.setattr("gptscan._normalize_and_filter_dirs", lambda x: x)
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("pathlib.Path.home", lambda: Path("/fake/home"))

    paths = gptscan.get_php_packages_paths()
    assert "/fake/home/.composer/vendor" in paths
    assert "/fake/home/.config/composer/vendor" in paths

def test_get_go_packages_paths_exception(monkeypatch):
    def mock_check_output(*args, **kwargs):
        raise RuntimeError("Go env failure")
    monkeypatch.setattr("subprocess.check_output", mock_check_output)
    monkeypatch.setattr("gptscan._normalize_and_filter_dirs", lambda x: x)
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("pathlib.Path.home", lambda: Path("/fake/home"))

    monkeypatch.delenv("GOPATH", raising=False)

    paths = gptscan.get_go_packages_paths()
    assert "/fake/home/go/pkg/mod" in paths
    assert "/fake/home/go/src" in paths

def test_set_scan_target_empty_or_no_textbox(monkeypatch):
    monkeypatch.setattr("gptscan.textbox", MagicMock())
    gptscan._set_scan_target("")

    monkeypatch.setattr("gptscan.textbox", None)
    gptscan._set_scan_target("some_path")

def test_set_scan_target_iterable(monkeypatch):
    mock_textbox = MagicMock()
    monkeypatch.setattr("gptscan.textbox", mock_textbox)
    mock_button = MagicMock()
    monkeypatch.setattr("gptscan.scan_button", mock_button)

    gptscan._set_scan_target(["/path1", "/path2"])

    mock_textbox.delete.assert_called_with(0, "end")
    insert_args = mock_textbox.insert.call_args[0]
    assert insert_args[0] == 0
    assert "/path1" in insert_args[1] and "/path2" in insert_args[1]
    mock_button.focus_set.assert_called_once()

def test_get_browser_bookmarks_snippets_sqlite_exception(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("gptscan.Path.home", lambda: tmp_path)

    ff_profile = tmp_path / ".mozilla" / "firefox" / "test.profile"
    ff_profile.mkdir(parents=True)
    db_path = ff_profile / "places.sqlite"
    db_path.write_text("not a sqlite database")

    snippets = gptscan.get_browser_bookmarks_snippets()
    assert snippets == []

def test_get_browser_bookmarks_snippets_json_exception(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("gptscan.Path.home", lambda: tmp_path)

    chrome_bookmarks = tmp_path / ".config" / "google-chrome" / "Default" / "Bookmarks"
    chrome_bookmarks.parent.mkdir(parents=True)
    chrome_bookmarks.write_text("corrupted invalid json contents")

    snippets = gptscan.get_browser_bookmarks_snippets()
    assert snippets == []
