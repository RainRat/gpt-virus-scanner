import json
import pytest
import io
import gptscan


def test_cli_import_directory(tmp_path, monkeypatch, capsys):
    """Test importing reports from a directory in CLI mode using run_cli."""
    dir_path = tmp_path / "reports"
    dir_path.mkdir()

    report1 = dir_path / "report1.json"
    data1 = [
        {
            "path": "script1.py",
            "own_conf": "80%",
            "admin_desc": "Dangerous eval found",
            "end-user_desc": "Suspicious code",
            "gpt_conf": "85%",
            "snippet": "eval(code)",
            "line": "12"
        }
    ]
    report1.write_text(json.dumps(data1), encoding="utf-8")

    report2 = dir_path / "report2.json"
    data2 = [
        {
            "path": "script2.js",
            "own_conf": "90%",
            "admin_desc": "Obfuscated payload",
            "end-user_desc": "Malware pattern",
            "gpt_conf": "95%",
            "snippet": "unescape('%20')",
            "line": "45"
        }
    ]
    report2.write_text(json.dumps(data2), encoding="utf-8")

    exit_code = gptscan.run_cli(
        targets=[],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        import_file=str(dir_path),
        quiet=True
    )

    captured = capsys.readouterr()
    imported_data = [json.loads(line) for line in captured.out.strip().splitlines() if line.strip()]

    assert len(imported_data) == 2
    paths = {item["path"] for item in imported_data}
    assert "script1.py" in paths
    assert "script2.js" in paths


def test_cli_import_url(monkeypatch, capsys):
    """Test importing report from a web link URL in CLI mode using run_cli."""
    url = "https://example.com/reports/scan.json"
    sample_json = json.dumps([
        {
            "path": "remote_script.py",
            "own_conf": "75%",
            "admin_desc": "Remote snippet issue",
            "end-user_desc": "Low risk",
            "gpt_conf": "70%",
            "snippet": "import socket",
            "line": "1"
        }
    ])

    monkeypatch.setattr(gptscan, "fetch_url_content", lambda u: sample_json.encode("utf-8"))

    exit_code = gptscan.run_cli(
        targets=[],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        output_format="csv",
        import_file=url,
        quiet=True
    )

    captured = capsys.readouterr()
    assert "remote_script.py" in captured.out
    assert "import socket" in captured.out
