import pytest
from gptscan import parse_triage_report, parse_report_content, strip_ansi

def test_strip_ansi():
    text_with_ansi = "\033[1;91mHIGH RISK\033[0m"
    assert strip_ansi(text_with_ansi) == "HIGH RISK"

    plain_text = "Just some standard text."
    assert strip_ansi(plain_text) == "Just some standard text."

def test_parse_plain_triage_report():
    report_content = """
--- GPT SCAN - CONSOLE TRIAGE REPORT (2 findings) ---

[1] HIGH RISK - test.py:123
    Local: 95%  AI: 85%  VT: https://virustotal.com/

    Admin: Highly suspicious eval instruction.

    User: This file evaluates untrusted input.

    > eval(user_input)

[2] MEDIUM RISK - script.sh
    Local: 60%

    Admin: Command execution found.

    > os.system("ls")
"""
    results = parse_triage_report(report_content)
    assert len(results) == 2

    # Check first finding
    f1 = results[0]
    assert f1["path"] == "test.py"
    assert f1["line"] == "123"
    assert f1["own_conf"] == "95%"
    assert f1["gpt_conf"] == "85%"
    assert f1["admin_desc"] == "Highly suspicious eval instruction."
    assert f1["end-user_desc"] == "This file evaluates untrusted input."
    assert f1["snippet"] == "eval(user_input)"

    # Check second finding
    f2 = results[1]
    assert f2["path"] == "script.sh"
    assert f2["line"] == "-"
    assert f2["own_conf"] == "60%"
    assert f2["gpt_conf"] == ""
    assert f2["admin_desc"] == "Command execution found."
    assert f2["end-user_desc"] == ""
    assert f2["snippet"] == 'os.system("ls")'


def test_parse_colorized_triage_report():
    # Use ANSI escape sequences
    report_content = """
\033[1m--- GPT SCAN - CONSOLE TRIAGE REPORT (1 finding) ---\033[0m

\033[0;90m[1]\033[0m \033[1;91mHIGH RISK\033[0m - \033[1mtest.py:10\033[0m
    \033[0;90mLocal:\033[0m \033[1;91m90%\033[0m  \033[0;90mAI:\033[0m \033[1;91m85%\033[0m

    \033[0;90mAdmin:\033[0m Line 1 of admin
    \033[0;90mAdmin:\033[0m Line 2 of admin

    \033[0;90mUser:\033[0m Line 1 of user

    \033[0;90m>\033[0m   preserved_spaces(x)
"""
    results = parse_triage_report(report_content)
    assert len(results) == 1

    f = results[0]
    assert f["path"] == "test.py"
    assert f["line"] == "10"
    assert f["own_conf"] == "90%"
    assert f["gpt_conf"] == "85%"
    assert f["admin_desc"] == "Line 1 of admin\nLine 2 of admin"
    assert f["end-user_desc"] == "Line 1 of user"
    assert f["snippet"] == "  preserved_spaces(x)"


def test_parse_report_content_integration():
    # Test text / log auto-detection
    report_content = """--- GPT SCAN - CONSOLE TRIAGE REPORT (1 finding) ---
[1] LOW RISK - safe.py
    Local: 10%

    > print("Hello")
"""

    # 1. Via filename hint .txt
    results_txt = parse_report_content(report_content, filename_hint="my_report.txt")
    assert len(results_txt) == 1
    assert results_txt[0]["path"] == "safe.py"

    # 2. Via filename hint .log
    results_log = parse_report_content(report_content, filename_hint="my_report.log")
    assert len(results_log) == 1

    # 3. Via auto-detection (no hint, but header present)
    results_auto = parse_report_content(report_content)
    assert len(results_auto) == 1
