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


def test_load_report_file_directory(tmp_path):
    """Verify load_report_file can load a directory recursively and combine findings from multiple files."""
    # Create directory structure
    dir_path = tmp_path / "baselines"
    dir_path.mkdir()

    sub_dir = dir_path / "nested"
    sub_dir.mkdir()

    # Create a JSON report
    json_report = dir_path / "report1.json"
    json_data = [
        {"path": "file1.py", "own_conf": "80%", "snippet": "eval(1)", "line": "10"}
    ]
    json_report.write_text(json.dumps(json_data), encoding="utf-8")

    # Create a CSV report inside the nested directory
    csv_report = sub_dir / "report2.csv"
    with open(csv_report, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet", "line"])
        writer.writerow(["file2.js", "70%", "Warning", "Warning", "", "exec('code')", "20"])

    # Create an unsupported extension file which should be ignored
    unsupported_file = dir_path / "report_ignored.txt_not_supported"
    unsupported_file.write_text("should be ignored", encoding="utf-8")

    # Load from the root baseline directory
    results = gptscan.load_report_file(str(dir_path))

    assert len(results) == 2
    # Verify both findings exist and are standardized
    paths = {r["path"] for r in results}
    assert "file1.py" in paths
    assert "file2.js" in paths


def test_baseline_filtering_cli_directory(tmp_path, monkeypatch):
    """Verify run_cli correctly loads findings from a baseline directory and bypasses them."""
    # Create directory with a couple of reports
    baseline_dir = tmp_path / "baseline_dir"
    baseline_dir.mkdir()

    json_report = baseline_dir / "rep1.json"
    json_data = [
        {"path": "known_unsafe.py", "own_conf": "90%", "snippet": "compromise()", "line": "5"}
    ]
    json_report.write_text(json.dumps(json_data), encoding="utf-8")

    # Mock findings where one is in baseline_dir (known_unsafe.py) and one is new (new_unsafe.py)
    mock_scan_results = [
        ('progress', (0, 2, "Scanning")),
        ('result', ("known_unsafe.py", "90%", "Critical", "Critical", "", "compromise()", "5")),
        ('result', ("new_unsafe.py", "85%", "Critical", "Critical", "", "shell_exec()", "15")),
        ('summary', (2, 1024, 0.2))
    ]

    monkeypatch.setattr(gptscan, "scan_files", lambda *args, **kwargs: mock_scan_results)

    output_file = tmp_path / "results.json"

    # Run run_cli specifying the directory as baseline_file
    threats = gptscan.run_cli(
        targets=["."],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        output_file=str(output_file),
        baseline_file=str(baseline_dir)
    )

    # Only new_unsafe.py should be printed/saved and counted as threat
    assert threats == 1

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    lines = content.strip().splitlines()
    assert len(lines) == 1
    result_record = json.loads(lines[0])
    assert result_record["path"] == "new_unsafe.py"
    assert result_record["snippet"] == "shell_exec()"
