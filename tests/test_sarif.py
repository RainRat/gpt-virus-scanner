import json
import pytest
from gptscan import generate_sarif, run_cli, Config

def test_generate_sarif_structure():
    """Verify the overall structure of the generated SARIF log."""
    results = []
    sarif = generate_sarif(results)

    assert sarif["version"] == "2.1.0"
    assert sarif["$schema"] == "https://json.schemastore.org/sarif-2.1.0.json"
    assert len(sarif["runs"]) == 1

    run = sarif["runs"][0]
    tool = run["tool"]["driver"]
    assert tool["name"] == "GPT Virus Scanner"
    assert len(tool["rules"]) > 0
    assert tool["rules"][0]["id"] == "GPTScan.MaliciousContent"
    assert sarif["runs"][0]["results"] == []

def test_generate_sarif_results_mapping():
    """Verify that scan results are correctly mapped to SARIF result objects."""
    results = [
        {
            "path": "/path/to/malicious.py",
            "own_conf": "95%",
            "gpt_conf": "99%",
            "snippet": "import os; os.system('rm -rf /')",
            "admin_desc": "Dangerous system call",
            "end-user_desc": "Deletes files"
        }
    ]

    sarif = generate_sarif(results)
    sarif_results = sarif["runs"][0]["results"]

    assert len(sarif_results) == 1
    res = sarif_results[0]

    assert res["ruleId"] == "GPTScan.MaliciousContent"
    assert res["level"] == "error"  # > 80%
    assert res["message"]["text"] == "Dangerous system call"

    location = res["locations"][0]["physicalLocation"]["artifactLocation"]
    assert location["uri"] == "/path/to/malicious.py"

    props = res["properties"]
    assert props["own_conf"] == "95%"
    assert props["gpt_conf"] == "99%"
    assert props["snippet"] == "import os; os.system('rm -rf /')"

def test_generate_sarif_confidence_levels():
    """Verify logic for mapping confidence percentages to SARIF levels."""
    # level mapping: >80 -> error, >50 -> warning, else -> note
    results = [
        {"path": "file1.py", "own_conf": "85%", "admin_desc": "High risk"},
        {"path": "file2.py", "own_conf": "60%", "admin_desc": "Medium risk"},
        {"path": "file3.py", "own_conf": "40%", "admin_desc": "Low risk"},
        {"path": "file4.py", "own_conf": "invalid", "admin_desc": "Unknown risk"},
    ]

    sarif = generate_sarif(results)
    sarif_results = sarif["runs"][0]["results"]

    assert len(sarif_results) == 4
    assert sarif_results[0]["level"] == "error"
    assert sarif_results[1]["level"] == "warning"
    assert sarif_results[2]["level"] == "note"
    assert sarif_results[3]["level"] == "note" # Default on error

def test_generate_sarif_path_normalization():
    """Verify that Windows paths are normalized to URI-compatible forward slashes."""
    results = [
        {"path": "C:\\Users\\User\\script.py", "own_conf": "90%", "admin_desc": "Test"}
    ]

    sarif = generate_sarif(results)
    uri = sarif["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]

    assert uri == "C:/Users/User/script.py"

def test_run_cli_output_sarif_format(monkeypatch, capsys):
    """Integration test for run_cli with output_format='sarif'."""
    import gptscan

    def mock_scan_files(*args, **kwargs):
        yield ('result', ("/path/file.py", "95%", "Admin Info", "User Info", "90%", "print('test')"))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    gptscan.run_cli(["/dummy"], False, True, False, 60, output_format='sarif')

    captured = capsys.readouterr()
    output = captured.out.strip()

    # Ensure output is valid JSON
    try:
        sarif_log = json.loads(output)
    except json.JSONDecodeError:
        pytest.fail("Output was not valid JSON")

    assert sarif_log["version"] == "2.1.0"
    assert len(sarif_log["runs"][0]["results"]) == 1
    assert sarif_log["runs"][0]["results"][0]["message"]["text"] == "Admin Info"
