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

    bookmarks_file = tmp_path / "Bookmarks"
    bookmarks_file.write_text(json.dumps(bookmarks_data))

    with patch("gptscan.Path.home", return_value=tmp_path):
        # We need to trick the platform check and discovery logic
        # For simplicity, let's mock bookmark_paths directly in get_browser_bookmarks_snippets
        # or mock the glob/exists calls.

        with patch("os.path.exists", side_effect=lambda p: str(p) == str(bookmarks_file) or os.path.exists(p)):
            with patch("gptscan.os.environ.get", return_value=str(tmp_path)):
                # Mocking bookmark_paths in the function is hard without changing the code.
                # Let's mock the list of paths directly.
                mock_paths = [(str(bookmarks_file), "Chrome")]
                with patch("gptscan.sys.platform", "linux"):
                    with patch("gptscan.get_browser_bookmarks_snippets", side_effect=None) as mock_func:
                        # Re-implementing discovery logic for the test to use our mock file
                        # Actually, let's just test the parsing logic by mocking bookmark_paths
                        pass

    # A better approach for unit testing this specific function:
    # Patch the discovery part to return our tmp file.

    with patch("gptscan.sys.platform", "linux"):
        with patch("gptscan.Path.home", return_value=tmp_path):
            with patch("gptscan.os.environ.get", return_value=str(tmp_path)):
                # This glob matches what's in gptscan.py for Linux Chrome
                # config = home / ".config"
                # chrome_base = config / "google-chrome"
                # bookmark_paths.append((str(chrome_base / "Default" / "Bookmarks"), "Chrome"))

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
