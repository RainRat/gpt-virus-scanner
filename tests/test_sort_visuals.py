import pytest
from unittest.mock import MagicMock
import gptscan

def test_sort_visual_indicators():
    # Create a mock for tv
    tv = MagicMock()
    tv.get_children.return_value = ["item1", "item2"]
    tv.set.side_effect = lambda k, c: "file1.py" if k == "item1" else "file2.py"

    # Mock columns attribute/dictionary behavior
    tv.__getitem__.side_effect = lambda key: ["path", "own_conf"] if key == "columns" else MagicMock()

    headings_text = {
        "path": "File Path",
        "own_conf": "Local Threat"
    }

    def mock_heading(col, option=None, text=None, command=None):
        if option == "text":
            return headings_text[col]
        if text is not None:
            headings_text[col] = text
        return MagicMock()

    tv.heading.side_effect = mock_heading

    # Initially, we sort by 'path' ascending (reverse=False)
    gptscan.sort_column(tv, "path", reverse=False)

    # Path should now have the ascending indicator
    assert headings_text["path"] == "File Path ▲"
    assert headings_text["own_conf"] == "Local Threat"

    # Sort by 'path' descending (reverse=True)
    gptscan.sort_column(tv, "path", reverse=True)
    assert headings_text["path"] == "File Path ▼"
    assert headings_text["own_conf"] == "Local Threat"

    # Sort by 'own_conf' ascending (reverse=False)
    gptscan.sort_column(tv, "own_conf", reverse=False)
    # Previous 'path' indicator should be cleared, and 'own_conf' should have the ascending indicator
    assert headings_text["path"] == "File Path"
    assert headings_text["own_conf"] == "Local Threat ▲"

def test_sort_visual_indicators_fallback():
    # If tv has no "columns" (e.g. raises TypeError or KeyError on tv["columns"]),
    # it should fall back to the standard list of columns and not crash
    tv = MagicMock()
    tv.get_children.return_value = ["item1", "item2"]
    tv.set.side_effect = lambda k, c: "1" if k == "item1" else "2"
    tv.__getitem__.side_effect = KeyError("no columns")

    headings_text = {
        "path": "File Path",
        "line": "Line",
        "own_conf": "Local Threat",
        "gpt_conf": "AI Threat",
        "admin_desc": "Admin Notes",
        "end-user_desc": "User Notes",
        "snippet": "Snippet"
    }

    def mock_heading(col, option=None, text=None, command=None):
        if option == "text":
            return headings_text.get(col, "")
        if text is not None:
            headings_text[col] = text
        return MagicMock()

    tv.heading.side_effect = mock_heading

    # Sort by 'line' descending (reverse=True)
    gptscan.sort_column(tv, "line", reverse=True)

    # Check that 'line' now has the descending indicator and others are clean
    assert headings_text["line"] == "Line ▼"
    assert headings_text["path"] == "File Path"
