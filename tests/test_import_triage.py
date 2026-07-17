import pytest
import os
import tempfile
from gptscan import parse_report_content, generate_console_report, load_report_file

def test_import_plain_triage_report():
    triage_content = """
--- GPT SCAN - CONSOLE TRIAGE REPORT (2 findings) ---

[1] HIGH RISK - /home/user/malicious.py:42
    Local: 85%  AI: 95%  VT: https://virustotal.com/

    Admin: Suspicious file execution.
    Contains dangerous calls.

    User: Potential threat found.

    > import os
    > os.system("rm -rf /")

[2] LOW RISK - src/safe.js
    Local: 15%

    Admin: Safe file.

    > console.log("hello");
"""

    results = parse_report_content(triage_content, filename_hint="report.txt")
    assert len(results) == 2

    # Verification for finding 1
    assert results[0]["path"] == "/home/user/malicious.py"
    assert results[0]["line"] == "42"
    assert results[0]["own_conf"] == "85%"
    assert results[0]["gpt_conf"] == "95%"
    assert results[0]["admin_desc"] == "Suspicious file execution.\nContains dangerous calls."
    assert results[0]["end-user_desc"] == "Potential threat found."
    assert results[0]["snippet"] == 'import os\nos.system("rm -rf /")'

    # Verification for finding 2
    assert results[1]["path"] == "src/safe.js"
    assert results[1]["line"] == "-"
    assert results[1]["own_conf"] == "15%"
    assert results[1]["gpt_conf"] == ""
    assert results[1]["admin_desc"] == "Safe file."
    assert results[1]["end-user_desc"] == ""
    assert results[1]["snippet"] == 'console.log("hello");'


def test_import_colorized_triage_report():
    # Colored with ANSI escape sequences
    triage_content = (
        "\033[1m--- GPT SCAN - CONSOLE TRIAGE REPORT (1 finding) ---\033[0m\n\n"
        "\033[0;90m[1]\033[0m \033[1;91mHIGH RISK\033[0m - \033[1m/path/to/script.py:10\033[0m\n"
        "    \033[0;90mLocal:\033[0m \033[1;91m90%\033[0m  \033[0;90mAI:\033[0m \033[1;91m95%\033[0m\n\n"
        "    \033[0;90mAdmin:\033[0m Dangerous code\n\n"
        "    \033[0;90m>\033[0m payload = 'evil'\n"
    )

    results = parse_report_content(triage_content, filename_hint="report.txt")
    assert len(results) == 1
    assert results[0]["path"] == "/path/to/script.py"
    assert results[0]["line"] == "10"
    assert results[0]["own_conf"] == "90%"
    assert results[0]["gpt_conf"] == "95%"
    assert results[0]["admin_desc"] == "Dangerous code"
    assert results[0]["snippet"] == "payload = 'evil'"


def test_triage_report_symmetry():
    # Verify that generating a console report and then importing it results in the same data
    original_results = [
        {
            "path": "test_file.py",
            "own_conf": "80%",
            "gpt_conf": "90%",
            "admin_desc": "Admin note line 1.\nAdmin note line 2.",
            "end-user_desc": "User note.",
            "snippet": "line 1\nline 2",
            "line": "15"
        }
    ]

    # Generate triage report (plain-text)
    report_text = generate_console_report(original_results, use_color=False)

    # Import triage report
    imported_results = parse_report_content(report_text, filename_hint="results.txt")
    assert len(imported_results) == 1

    assert imported_results[0]["path"] == original_results[0]["path"]
    assert imported_results[0]["line"] == original_results[0]["line"]
    assert imported_results[0]["own_conf"] == original_results[0]["own_conf"]
    assert imported_results[0]["gpt_conf"] == original_results[0]["gpt_conf"]
    assert imported_results[0]["admin_desc"] == original_results[0]["admin_desc"]
    assert imported_results[0]["end-user_desc"] == original_results[0]["end-user_desc"]

    # Note: the snippet printed in the triage report has a limit of 3 lines, which our snippet matches.
    # The snippet line headers are stripped and restored.
    assert imported_results[0]["snippet"] == original_results[0]["snippet"]


def test_import_triage_file_loading():
    triage_content = """
[1] HIGH RISK - script.py:1
    Local: 100%

    Admin: Critical threat.

    > print("harm")
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "report.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(triage_content)

        results = load_report_file(file_path)
        assert len(results) == 1
        assert results[0]["path"] == "script.py"
        assert results[0]["line"] == "1"
        assert results[0]["own_conf"] == "100%"
        assert results[0]["admin_desc"] == "Critical threat."
        assert results[0]["snippet"] == 'print("harm")'
