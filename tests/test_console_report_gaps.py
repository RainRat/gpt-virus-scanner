import pytest
import shutil
from unittest.mock import MagicMock, patch
from gptscan import generate_console_report, Config

def test_generate_console_report_empty():
    assert generate_console_report([]) == "No findings to report."

def test_generate_console_report_medium_risk(monkeypatch):
    monkeypatch.setattr(Config, 'THRESHOLD', 50)
    results = [
        {
            "path": "medium.py",
            "own_conf": "60%",
            "admin_desc": "Some risk",
            "snippet": "code()"
        }
    ]
    report = generate_console_report(results, use_color=False)
    assert "MEDIUM RISK" in report
    assert "medium.py" in report
    assert "Local: 60%" in report

def test_generate_console_report_no_line_num():
    results = [
        {
            "path": "file.py",
            "own_conf": "90%",
            "snippet": "code()"
        }
    ]
    report = generate_console_report(results, use_color=False)
    # When line is missing, it should just show the path
    assert "HIGH RISK - file.py" in report
    assert "file.py:-" not in report

def test_generate_console_report_snippet_truncation():
    results = [
        {
            "path": "test.py",
            "own_conf": "90%",
            "snippet": "A" * 100
        }
    ]

    # Mock terminal width to 40 columns
    # max_snippet_len = max(20, 40 - 8) = 32
    # Truncation: sl[:32-3] + "..." = sl[:29] + "..."
    with patch("shutil.get_terminal_size", return_value=MagicMock(columns=40)):
        report = generate_console_report(results, use_color=False)
        expected_snippet = "A" * 29 + "..."
        assert f"> {expected_snippet}" in report

def test_generate_console_report_snippet_line_limit():
    results = [
        {
            "path": "test.py",
            "own_conf": "90%",
            "snippet": "line1\nline2\nline3\nline4\nline5"
        }
    ]
    report = generate_console_report(results, use_color=False)
    assert "> line1" in report
    assert "> line2" in report
    assert "> line3" in report
    assert "> line4" not in report

def test_generate_console_report_multiline_ai_notes():
    results = [
        {
            "path": "test.py",
            "own_conf": "90%",
            "admin_desc": "Admin line 1\nAdmin line 2",
            "end-user_desc": "User line 1\nUser line 2",
            "snippet": "code()"
        }
    ]
    report = generate_console_report(results, use_color=False)
    assert "Admin: Admin line 1" in report
    assert "Admin: Admin line 2" in report
    assert "User: User line 1" in report
    assert "User: User line 2" in report

def test_generate_console_report_only_user_notes():
    results = [
        {
            "path": "test.py",
            "own_conf": "90%",
            "end-user_desc": "Only user notes",
            "snippet": "code()"
        }
    ]
    report = generate_console_report(results, use_color=False)
    assert "Admin:" not in report
    assert "User: Only user notes" in report

def test_generate_console_report_color_conf_logic():
    # Test the inner color_conf function branches
    results = [
        {"path": "high.py", "own_conf": "85%", "snippet": "c()"},
        {"path": "med.py", "own_conf": "60%", "snippet": "c()"},
        {"path": "low.py", "own_conf": "10%", "snippet": "c()"}
    ]

    report = generate_console_report(results, use_color=True)

    # High risk color (RED + BOLD)
    assert "\033[1;91m\033[1m85%\033[0m" in report
    # Medium risk color (YELLOW + BOLD)
    assert "\033[1;93m\033[1m60%\033[0m" in report
    # Low risk (no extra color in color_conf)
    assert "10%" in report
    assert "\033[1;91m\033[1m10%\033[0m" not in report
