import json
import io
import csv
import sys
import pytest
from unittest.mock import patch, MagicMock

import gptscan


def test_cli_stats_default_json_output(tmp_path, monkeypatch):
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')", encoding='utf-8')

    out_stream = io.StringIO()
    monkeypatch.setattr(gptscan, "get_model", lambda: MagicMock())

    result_code = gptscan.run_cli(
        targets=[str(test_file)],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_format='report',
        dry_run=True,
        quiet=True,
        stats_only=True,
    )

    # run_cli returns integer threats found
    assert result_code == 0


def test_cli_stats_json_payload(tmp_path, monkeypatch):
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')", encoding='utf-8')

    out_stream = io.StringIO()
    with patch('sys.stdout', out_stream):
        gptscan.run_cli(
            targets=[str(test_file)],
            deep=False,
            show_all=False,
            use_gpt=False,
            rate_limit=60,
            output_format='json',
            dry_run=True,
            quiet=True,
            stats_only=True,
        )

    output = out_stream.getvalue().strip()
    data = json.loads(output)
    assert "scanned_files" in data
    assert "threats_found" in data
    assert "high_risk" in data
    assert "medium_risk" in data
    assert "scanned_bytes" in data
    assert "elapsed_seconds" in data
    assert data["scanned_files"] == 1
    assert data["threats_found"] == 0


def test_cli_stats_csv_output(tmp_path, monkeypatch):
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')", encoding='utf-8')

    out_stream = io.StringIO()
    with patch('sys.stdout', out_stream):
        gptscan.run_cli(
            targets=[str(test_file)],
            deep=False,
            show_all=False,
            use_gpt=False,
            rate_limit=60,
            output_format='csv',
            dry_run=True,
            quiet=True,
            stats_only=True,
        )

    output = out_stream.getvalue().strip().splitlines()
    assert len(output) == 2
    reader = csv.reader(output)
    header = next(reader)
    row = next(reader)
    assert "scanned_files" in header
    assert "threats_found" in header


def test_cli_summary_only_json(tmp_path):
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')", encoding='utf-8')

    out_stream = io.StringIO()
    with patch('sys.stdout', out_stream):
        gptscan.run_cli(
            targets=[str(test_file)],
            deep=False,
            show_all=False,
            use_gpt=False,
            rate_limit=60,
            output_format='json',
            dry_run=True,
            quiet=True,
            summary_only=True,
        )

    output = out_stream.getvalue().strip()
    data = json.loads(output)
    assert data["scanned_files"] == 1
    assert data["threats_found"] == 0


def test_cli_stats_with_baseline(tmp_path, monkeypatch):
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')", encoding='utf-8')
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text("[]", encoding='utf-8')

    out_stream = io.StringIO()
    with patch('sys.stdout', out_stream):
        gptscan.run_cli(
            targets=[str(test_file)],
            deep=False,
            show_all=False,
            use_gpt=False,
            rate_limit=60,
            output_format='json',
            dry_run=True,
            quiet=True,
            stats_only=True,
            baseline_file=str(baseline_file),
        )

    output = out_stream.getvalue().strip()
    data = json.loads(output)
    assert "bypassed_baseline" in data


def test_main_cli_stats_args(monkeypatch):
    test_args = ["gptscan.py", ".", "--cli", "--stats", "--dry-run", "-q"]
    monkeypatch.setattr(sys, "argv", test_args)

    run_cli_mock = MagicMock(return_value=0)
    monkeypatch.setattr(gptscan, "run_cli", run_cli_mock)

    gptscan.main()

    run_cli_mock.assert_called_once()
    assert run_cli_mock.call_args.kwargs.get("stats_only") is True
