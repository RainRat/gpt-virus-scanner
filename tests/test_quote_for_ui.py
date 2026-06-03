import sys
from unittest.mock import patch
import gptscan

def test_quote_for_ui_posix_unchanged_without_special_chars():
    with patch("gptscan.sys.platform", "linux"):
        assert gptscan._quote_for_ui("/path/to/file") == "/path/to/file"

def test_quote_for_ui_posix_quoted_with_spaces():
    with patch("gptscan.sys.platform", "linux"):
        result = gptscan._quote_for_ui("/path with space/file")
        assert result.startswith("'") and result.endswith("'")
        assert "/path with space/file" in result

def test_quote_for_ui_windows_unchanged_without_special_chars():
    with patch("gptscan.sys.platform", "win32"):
        assert gptscan._quote_for_ui("C:\\path\\to\\file") == "C:\\path\\to\\file"

def test_quote_for_ui_windows_double_quoted_with_spaces():
    with patch("gptscan.sys.platform", "win32"):
        assert gptscan._quote_for_ui("C:\\path with space\\file") == '"C:\\path with space\\file"'

def test_quote_for_ui_windows_escaped_with_double_quotes():
    with patch("gptscan.sys.platform", "win32"):
        assert gptscan._quote_for_ui('C:\\path"with"quotes') == '"C:\\path""with""quotes"'

def test_quote_for_ui_windows_quoted_with_special_shell_chars():
    with patch("gptscan.sys.platform", "win32"):
        assert gptscan._quote_for_ui("C:\\path%with&special^chars|and<more>") == '"C:\\path%with&special^chars|and<more>"'

def test_quote_for_ui_windows_unchanged_with_single_quotes():
    with patch("gptscan.sys.platform", "win32"):
        assert gptscan._quote_for_ui("C:\\path'with'single'quotes") == "C:\\path'with'single'quotes"

def test_quote_for_ui_windows_individual_triggers():
    with patch("gptscan.sys.platform", "win32"):
        assert gptscan._quote_for_ui("path ") == '"path "'
        assert gptscan._quote_for_ui('path"') == '"path"""'
        assert gptscan._quote_for_ui("path%") == '"path%"'
        assert gptscan._quote_for_ui("path&") == '"path&"'
        assert gptscan._quote_for_ui("path^") == '"path^"'
        assert gptscan._quote_for_ui("path|") == '"path|"'
        assert gptscan._quote_for_ui("path<") == '"path<"'
        assert gptscan._quote_for_ui("path>") == '"path>"'
