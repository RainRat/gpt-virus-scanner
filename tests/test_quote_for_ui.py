import sys
from unittest.mock import patch
import gptscan

def test_quote_for_ui_posix_simple():
    with patch("sys.platform", "linux"):
        assert gptscan._quote_for_ui("path/to/file") == "path/to/file"

def test_quote_for_ui_posix_with_space():
    with patch("sys.platform", "linux"):
        assert gptscan._quote_for_ui("path with space") == "'path with space'"

def test_quote_for_ui_windows_simple():
    with patch("sys.platform", "win32"):
        assert gptscan._quote_for_ui(r"C:\path\to\file") == r"C:\path\to\file"

def test_quote_for_ui_windows_with_space():
    with patch("sys.platform", "win32"):
        assert gptscan._quote_for_ui(r"C:\path to\file") == r'"C:\path to\file"'

def test_quote_for_ui_windows_with_special_chars():
    with patch("sys.platform", "win32"):
        assert gptscan._quote_for_ui(r"C:\path%with&special^chars|and<others>") == r'"C:\path%with&special^chars|and<others>"'

def test_quote_for_ui_windows_escapes_double_quotes():
    with patch("sys.platform", "win32"):
        path = r'C:\path "with" quotes'
        expected = r'"C:\path ""with"" quotes"'
        assert gptscan._quote_for_ui(path) == expected

def test_quote_for_ui_windows_only_quotes_if_needed():
    with patch("sys.platform", "win32"):
        path = r"C:\Windows\System32\cmd.exe"
        assert gptscan._quote_for_ui(path) == path

def test_quote_for_ui_windows_quotes_and_escapes_single_quote_path():
    with patch("sys.platform", "win32"):
        path = '"'
        expected = '""""'
        assert gptscan._quote_for_ui(path) == expected
