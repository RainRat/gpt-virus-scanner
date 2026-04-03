import pytest
from gptscan import _clean_snippet_for_ai

def test_clean_snippet_basic():
    snippet = "print('hello')\n"
    expected = "print('hello')"
    assert _clean_snippet_for_ai(snippet) == expected

def test_clean_snippet_leading_trailing_whitespace():
    snippet = "   \nprint('hello')\n   "
    expected = "print('hello')"
    assert _clean_snippet_for_ai(snippet) == expected

def test_clean_snippet_empty_lines():
    snippet = "line1\n\nline2\n\n\nline3"
    expected = "line1\nline2\nline3"
    assert _clean_snippet_for_ai(snippet) == expected

def test_clean_snippet_whitespace_only_lines():
    snippet = "line1\n   \nline2\n\t\nline3"
    expected = "line1\nline2\nline3"
    assert _clean_snippet_for_ai(snippet) == expected

def test_clean_snippet_complex():
    snippet = """

    line1

    line2

    line3

    """
    expected = "line1\n    line2\n    line3"
    assert _clean_snippet_for_ai(snippet) == expected

def test_clean_snippet_no_content():
    assert _clean_snippet_for_ai("   \n   \n   ") == ""
    assert _clean_snippet_for_ai("") == ""
