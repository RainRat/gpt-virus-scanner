import json
import os
import sys
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from gptscan import get_browser_bookmarks_snippets

def test_get_browser_bookmarks_snippets_chromium(tmp_path):
    # Mock Chromium Bookmarks file
    bookmarks_data = {
        "roots": {
            "bookmark_bar": {
                "children": [
                    {
                        "name": "Normal URL",
                        "type": "url",
                        "url": "https://google.com"
                    },
                    {
                        "name": "Malicious Bookmarklet",
                        "type": "url",
                        "url": "javascript:alert('XSS')"
                    },
                    {
                        "name": "Folder",
                        "type": "folder",
                        "children": [
                            {
                                "name": "Data URL",
                                "type": "url",
                                "url": "data:text/html,<script>alert(1)</script>"
                            }
                        ]
                    }
                ],
                "type": "folder"
            }
        }
    }

    with patch("gptscan.sys.platform", "linux"):
        with patch("gptscan.Path.home", return_value=tmp_path):
            with patch("gptscan.os.environ.get", return_value=str(tmp_path)):
                chrome_bookmarks = tmp_path / ".config" / "google-chrome" / "Default" / "Bookmarks"
                chrome_bookmarks.parent.mkdir(parents=True)
                chrome_bookmarks.write_text(json.dumps(bookmarks_data))

                snippets = get_browser_bookmarks_snippets()

                titles = [s[0] for s in snippets]
                contents = [s[1].decode('utf-8') for s in snippets]

                assert "[Chrome Bookmark] Malicious Bookmarklet" in titles
                assert "javascript:alert('XSS')" in contents
                assert "[Chrome Bookmark] Data URL" in titles
                assert "data:text/html,<script>alert(1)</script>" in contents
                assert len(snippets) == 2

def test_get_browser_bookmarks_snippets_firefox(tmp_path):
    # Mock Firefox places.sqlite
    ff_profile = tmp_path / ".mozilla" / "firefox" / "test.profile"
    ff_profile.mkdir(parents=True)
    db_path = ff_profile / "places.sqlite"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE moz_places (id INTEGER PRIMARY KEY, url TEXT)")
    cursor.execute("CREATE TABLE moz_bookmarks (id INTEGER PRIMARY KEY, fk INTEGER, title TEXT)")

    cursor.execute("INSERT INTO moz_places (url) VALUES ('https://google.com')")
    cursor.execute("INSERT INTO moz_bookmarks (fk, title) VALUES (1, 'Google')")

    cursor.execute("INSERT INTO moz_places (url) VALUES ('javascript:alert(\"FF\")')")
    cursor.execute("INSERT INTO moz_bookmarks (fk, title) VALUES (2, 'FF Script')")

    cursor.execute("INSERT INTO moz_places (url) VALUES ('data:text/plain,secret')")
    cursor.execute("INSERT INTO moz_bookmarks (fk, title) VALUES (3, 'FF Data')")

    conn.commit()
    conn.close()

    with patch("gptscan.sys.platform", "linux"):
        with patch("gptscan.Path.home", return_value=tmp_path):
            snippets = get_browser_bookmarks_snippets()

            titles = [s[0] for s in snippets]
            contents = [s[1].decode('utf-8') for s in snippets]

            assert "[Firefox Bookmark] FF Script" in titles
            assert "javascript:alert(\"FF\")" in contents
            assert "[Firefox Bookmark] FF Data" in titles
            assert "data:text/plain,secret" in contents
            assert len(snippets) == 2

def test_get_browser_bookmarks_snippets_windows(tmp_path, monkeypatch):
    local_appdata = tmp_path / "LocalAppData"
    appdata = tmp_path / "AppData"
    local_appdata.mkdir(parents=True)
    appdata.mkdir(parents=True)

    # 1. Chrome Bookmarks
    chrome_bookmarks = local_appdata / "Google" / "Chrome" / "User Data" / "Default" / "Bookmarks"
    chrome_bookmarks.parent.mkdir(parents=True)
    bookmarks_data = {
        "roots": {
            "bookmark_bar": {
                "type": "folder",
                "children": [
                    {
                        "name": "Windows Chrome Bookmarklet",
                        "type": "url",
                        "url": "javascript:alert('WinChrome')"
                    }
                ]
            }
        }
    }
    chrome_bookmarks.write_text(json.dumps(bookmarks_data), encoding="utf-8")

    # 2. Firefox bookmarks (places.sqlite)
    ff_profile = appdata / "Mozilla" / "Firefox" / "Profiles" / "test.profile"
    ff_profile.mkdir(parents=True)
    db_path = ff_profile / "places.sqlite"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE moz_places (id INTEGER PRIMARY KEY, url TEXT)")
    cursor.execute("CREATE TABLE moz_bookmarks (id INTEGER PRIMARY KEY, fk INTEGER, title TEXT)")
    cursor.execute("INSERT INTO moz_places (url) VALUES ('javascript:alert(\"WinFF\")')")
    cursor.execute("INSERT INTO moz_bookmarks (fk, title) VALUES (1, 'WinFF Script')")
    conn.commit()
    conn.close()

    # Apply patches
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))
    monkeypatch.setenv("APPDATA", str(appdata))

    snippets = get_browser_bookmarks_snippets()

    titles = [s[0] for s in snippets]
    contents = [s[1].decode('utf-8') for s in snippets]

    assert "[Chrome Bookmark] Windows Chrome Bookmarklet" in titles
    assert "javascript:alert('WinChrome')" in contents
    assert "[Firefox Bookmark] WinFF Script" in titles
    assert "javascript:alert(\"WinFF\")" in contents

def test_get_browser_bookmarks_snippets_darwin(tmp_path, monkeypatch):
    lib_support = tmp_path / "Library" / "Application Support"
    lib_support.mkdir(parents=True)

    # 1. Edge Bookmarks
    edge_bookmarks = lib_support / "Microsoft Edge" / "Default" / "Bookmarks"
    edge_bookmarks.parent.mkdir(parents=True)
    bookmarks_data = {
        "roots": {
            "bookmark_bar": {
                "type": "folder",
                "children": [
                    {
                        "name": "Mac Edge Bookmarklet",
                        "type": "url",
                        "url": "javascript:alert('MacEdge')"
                    }
                ]
            }
        }
    }
    edge_bookmarks.write_text(json.dumps(bookmarks_data), encoding="utf-8")

    # 2. Firefox bookmarks (places.sqlite)
    ff_profile = lib_support / "Firefox" / "Profiles" / "mac.profile"
    ff_profile.mkdir(parents=True)
    db_path = ff_profile / "places.sqlite"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE moz_places (id INTEGER PRIMARY KEY, url TEXT)")
    cursor.execute("CREATE TABLE moz_bookmarks (id INTEGER PRIMARY KEY, fk INTEGER, title TEXT)")
    cursor.execute("INSERT INTO moz_places (url) VALUES ('javascript:alert(\"MacFF\")')")
    cursor.execute("INSERT INTO moz_bookmarks (fk, title) VALUES (1, 'MacFF Script')")
    conn.commit()
    conn.close()

    # Apply patches
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    snippets = get_browser_bookmarks_snippets()

    titles = [s[0] for s in snippets]
    contents = [s[1].decode('utf-8') for s in snippets]

    assert "[Edge Bookmark] Mac Edge Bookmarklet" in titles
    assert "javascript:alert('MacEdge')" in contents
    assert "[Firefox Bookmark] MacFF Script" in titles
    assert "javascript:alert(\"MacFF\")" in contents

def test_get_browser_bookmarks_snippets_multiprofile_discovery(tmp_path, monkeypatch):
    local_appdata = tmp_path / "LocalAppData"
    local_appdata.mkdir(parents=True)

    profile_bookmarks = local_appdata / "Google" / "Chrome" / "User Data" / "Profile 1" / "Bookmarks"
    profile_bookmarks.parent.mkdir(parents=True)

    bookmarks_data = {
        "roots": {
            "bookmark_bar": {
                "type": "folder",
                "children": [
                    {
                        "name": "Profile 1 Bookmarklet",
                        "type": "url",
                        "url": "javascript:alert('Profile1')"
                    }
                ]
            }
        }
    }
    profile_bookmarks.write_text(json.dumps(bookmarks_data), encoding="utf-8")

    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))

    snippets = get_browser_bookmarks_snippets()

    titles = [s[0] for s in snippets]
    contents = [s[1].decode('utf-8') for s in snippets]

    assert "[Chrome Bookmark] Profile 1 Bookmarklet" in titles
    assert "javascript:alert('Profile1')" in contents

def test_get_browser_bookmarks_snippets_untitled_and_edge_cases(tmp_path, monkeypatch):
    # Test Firefox NULL title and Chromium missing name property
    config_dir = tmp_path / ".config"
    config_dir.mkdir(parents=True)

    # 1. Chromium with missing name attribute
    chrome_bookmarks = config_dir / "google-chrome" / "Default" / "Bookmarks"
    chrome_bookmarks.parent.mkdir(parents=True)
    bookmarks_data = {
        "roots": {
            "other": {
                "type": "folder",
                "children": [
                    {
                        "type": "url",
                        "url": "javascript:console.log('NoName')"
                    }
                ]
            }
        }
    }
    chrome_bookmarks.write_text(json.dumps(bookmarks_data), encoding="utf-8")

    # 2. Firefox with NULL title
    ff_profile = tmp_path / ".mozilla" / "firefox" / "null_title.profile"
    ff_profile.mkdir(parents=True)
    db_path = ff_profile / "places.sqlite"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE moz_places (id INTEGER PRIMARY KEY, url TEXT)")
    cursor.execute("CREATE TABLE moz_bookmarks (id INTEGER PRIMARY KEY, fk INTEGER, title TEXT)")
    cursor.execute("INSERT INTO moz_places (url) VALUES ('javascript:alert(\"NullTitle\")')")
    cursor.execute("INSERT INTO moz_bookmarks (fk, title) VALUES (1, NULL)")
    conn.commit()
    conn.close()

    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    snippets = get_browser_bookmarks_snippets()

    titles = [s[0] for s in snippets]
    contents = [s[1].decode('utf-8') for s in snippets]

    assert "[Chrome Bookmark] Untitled" in titles
    assert "javascript:console.log('NoName')" in contents
    assert "[Firefox Bookmark] Untitled" in titles
    assert "javascript:alert(\"NullTitle\")" in contents

def test_get_browser_bookmarks_snippets_corrupted_files(tmp_path, monkeypatch):
    config_dir = tmp_path / ".config"
    config_dir.mkdir(parents=True)

    # Corrupted JSON for Chromium
    chrome_bookmarks = config_dir / "google-chrome" / "Default" / "Bookmarks"
    chrome_bookmarks.parent.mkdir(parents=True)
    chrome_bookmarks.write_text("{ invalid json content: ", encoding="utf-8")

    # Corrupted SQLite for Firefox
    ff_profile = tmp_path / ".mozilla" / "firefox" / "corrupt.profile"
    ff_profile.mkdir(parents=True)
    db_path = ff_profile / "places.sqlite"
    db_path.write_bytes(b"NOT A SQLITE DATABASE")

    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Should handle both corrupted files gracefully without raising exception
    snippets = get_browser_bookmarks_snippets()
    assert snippets == []
