import os
import json
import csv
import pytest
from pathlib import Path
import gptscan


def test_get_finding_signature():
    """Verify that get_finding_signature correctly normalizes paths and formats identifiers."""
    # Test path normalization and snippet matching
    item_1 = {
        "path": "folder\\sub/file.py",
        "snippet": "  print('hello')  ",
        "line": "12"
    }
    sig_1 = gptscan.get_finding_signature(item_1)
    # Expected normalized path and stripped snippet
    assert sig_1 == ("folder/sub/file.py", "print('hello')")

    # Test empty snippet fallback to line number
    item_2 = {
        "path": "folder/sub/file.py",
        "snippet": "",
        "line": "42"
    }
    sig_2 = gptscan.get_finding_signature(item_2)
    assert sig_2 == ("folder/sub/file.py", "__line__:42")

    # Test empty snippet and line number fallback
    item_3 = {
        "path": "folder/sub/file.py",
    }
    sig_3 = gptscan.get_finding_signature(item_3)
    assert sig_3 == ("folder/sub/file.py", "__line__:-")


def test_baseline_filtering_cli_json_format(tmp_path, monkeypatch):
    """Verify that run_cli correctly loads a JSON baseline and filters out matching findings."""
    # Create baseline file
    baseline_file = tmp_path / "baseline.json"
    baseline_data = [
        {
            "path": "unsafe.py",
            "own_conf": "60%",
            "gpt_conf": "",
            "admin_desc": "Known alert",
            "end-user_desc": "Known alert for user",
            "snippet": "os.system('rm -rf /')",
            "line": "5"
        }
    ]
    baseline_file.write_text(json.dumps(baseline_data), encoding="utf-8")

    # Define mock findings that scan_files generator would produce
    # Yields two findings: one existing in the baseline (unsafe.py), one new (dangerous.py)
    mock_scan_results = [
        ('progress', (0, 2, "Scanning")),
        ('result', ("unsafe.py", "60%", "Known alert", "Known alert for user", "", "os.system('rm -rf /')", "5")),
        ('result', ("dangerous.py", "75%", "New alert", "New alert for user", "", "eval(userInput)", "10")),
        ('summary', (2, 2048, 0.5))
    ]

    monkeypatch.setattr(gptscan, "scan_files", lambda *args, **kwargs: mock_scan_results)

    # Output file for scan results
    output_file = tmp_path / "results.json"

    # Run run_cli
    threats = gptscan.run_cli(
        targets=["."],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        output_file=str(output_file),
        baseline_file=str(baseline_file)
    )

    # Since unsafe.py was in the baseline, it should have been bypassed.
    # Only dangerous.py should be counted as a threat.
    assert threats == 1

    # Verify output file content
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    lines = content.strip().splitlines()
    assert len(lines) == 1
    result_record = json.loads(lines[0])
    assert result_record["path"] == "dangerous.py"
    assert result_record["snippet"] == "eval(userInput)"


def test_baseline_filtering_cli_csv_format(tmp_path, monkeypatch):
    """Verify that run_cli correctly loads a CSV baseline and filters out matching findings."""
    # Create a CSV baseline file
    baseline_file = tmp_path / "baseline.csv"
    with open(baseline_file, "w", newline="", encoding="utf-8") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(["path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet", "line"])
        writer.writerow(["safe.py", "55%", "Alert 1", "Alert 1", "", "os.popen('cat')", "2"])

    # Define mock scan results containing only safe.py (the one in the baseline)
    mock_scan_results = [
        ('progress', (0, 1, "Scanning")),
        ('result', ("safe.py", "55%", "Alert 1", "Alert 1", "", "os.popen('cat')", "2")),
        ('summary', (1, 1024, 0.1))
    ]

    monkeypatch.setattr(gptscan, "scan_files", lambda *args, **kwargs: mock_scan_results)

    output_file = tmp_path / "results.csv"

    threats = gptscan.run_cli(
        targets=["."],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_format="csv",
        output_file=str(output_file),
        baseline_file=str(baseline_file)
    )

    # Since the only finding was in the baseline, it is bypassed completely
    assert threats == 0

    assert output_file.exists()
    with open(output_file, "r", encoding="utf-8") as csv_f:
        reader = csv.reader(csv_f)
        rows = list(reader)
    # Header should exist, but no other rows
    assert len(rows) == 1
    assert rows[0][0] == "path"


def test_baseline_fail_threshold_integration(tmp_path, monkeypatch):
    """Verify that --fail-threshold does not trigger if all matching threats are bypassed by baseline."""
    # Create baseline file
    baseline_file = tmp_path / "baseline.json"
    baseline_data = [
        {
            "path": "unsafe.py",
            "own_conf": "90%",
            "snippet": "system_compromise()",
            "line": "1"
        }
    ]
    baseline_file.write_text(json.dumps(baseline_data), encoding="utf-8")

    # Mock finding of unsafe.py (90% threat level)
    mock_scan_results = [
        ('progress', (0, 1, "Scanning")),
        ('result', ("unsafe.py", "90%", "Critical threat", "Critical threat", "", "system_compromise()", "1")),
        ('summary', (1, 512, 0.1))
    ]

    monkeypatch.setattr(gptscan, "scan_files", lambda *args, **kwargs: mock_scan_results)

    # If we set fail_threshold to 80, the 90% unsafe.py normally triggers a failure,
    # but since it is in baseline, threats returned should be 0, so no failure is triggered.
    threats = gptscan.run_cli(
        targets=["."],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        fail_threshold=80,
        baseline_file=str(baseline_file)
    )

    assert threats == 0


def test_baseline_missing_file_exits(capsys, monkeypatch):
    """Verify that specifying a non-existent baseline file prints an error and exits."""
    # Mock sys.exit to check if it exits
    exit_called = False
    def mock_exit(code):
        nonlocal exit_called
        exit_called = True
        raise SystemExit(code)

    monkeypatch.setattr(gptscan.sys, "exit", mock_exit)

    with pytest.raises(SystemExit):
        gptscan.run_cli(
            targets=["."],
            deep=False,
            show_all=False,
            use_gpt=False,
            rate_limit=60,
            baseline_file="non_existent_baseline_file.json"
        )

    assert exit_called
    captured = capsys.readouterr()
    assert "Error loading baseline file non_existent_baseline_file.json" in captured.err or "Error loading baseline file non_existent_baseline_file.json" in captured.out
