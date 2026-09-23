import json
import sys
from unittest.mock import MagicMock
import pytest
import gptscan


def test_reverse_sort_path(tmp_path, capsys):
    import_report = tmp_path / "report.json"
    data = [
        {"path": "b_file.py", "line": "10", "own_conf": "50%", "admin_desc": "", "end-user_desc": "", "gpt_conf": "", "snippet": "b"},
        {"path": "a_file.py", "line": "5", "own_conf": "50%", "admin_desc": "", "end-user_desc": "", "gpt_conf": "", "snippet": "a"},
    ]
    import_report.write_text(json.dumps(data), encoding="utf-8")

    # Path ascending (default)
    gptscan.run_cli([], deep=False, show_all=True, use_gpt=False, rate_limit=60, output_format="csv", quiet=True, import_file=str(import_report), sort_by="path", reverse_sort=False)
    out1 = capsys.readouterr().out
    lines1 = [line.split(",")[0] for line in out1.strip().splitlines() if line]
    assert lines1 == ["path", "a_file.py", "b_file.py"]

    # Path reversed (descending)
    gptscan.run_cli([], deep=False, show_all=True, use_gpt=False, rate_limit=60, output_format="csv", quiet=True, import_file=str(import_report), sort_by="path", reverse_sort=True)
    out2 = capsys.readouterr().out
    lines2 = [line.split(",")[0] for line in out2.strip().splitlines() if line]
    assert lines2 == ["path", "b_file.py", "a_file.py"]


def test_reverse_sort_line(tmp_path, capsys):
    import_report = tmp_path / "report.json"
    data = [
        {"path": "file.py", "line": "20", "own_conf": "50%", "admin_desc": "", "end-user_desc": "", "gpt_conf": "", "snippet": "20"},
        {"path": "file.py", "line": "5", "own_conf": "50%", "admin_desc": "", "end-user_desc": "", "gpt_conf": "", "snippet": "5"},
    ]
    import_report.write_text(json.dumps(data), encoding="utf-8")

    # Line ascending
    gptscan.run_cli([], deep=False, show_all=True, use_gpt=False, rate_limit=60, output_format="csv", quiet=True, import_file=str(import_report), sort_by="line", reverse_sort=False)
    out1 = capsys.readouterr().out
    lines1 = [line.split(",")[-1] for line in out1.strip().splitlines() if line]
    assert lines1 == ["line", "5", "20"]

    # Line reversed (descending)
    gptscan.run_cli([], deep=False, show_all=True, use_gpt=False, rate_limit=60, output_format="csv", quiet=True, import_file=str(import_report), sort_by="line", reverse_sort=True)
    out2 = capsys.readouterr().out
    lines2 = [line.split(",")[-1] for line in out2.strip().splitlines() if line]
    assert lines2 == ["line", "20", "5"]


def test_reverse_sort_threat(tmp_path, capsys):
    import_report = tmp_path / "report.json"
    data = [
        {"path": "low.py", "line": "1", "own_conf": "20%", "admin_desc": "", "end-user_desc": "", "gpt_conf": "", "snippet": "low"},
        {"path": "high.py", "line": "1", "own_conf": "90%", "admin_desc": "", "end-user_desc": "", "gpt_conf": "", "snippet": "high"},
    ]
    import_report.write_text(json.dumps(data), encoding="utf-8")

    # Default threat sorting: highest first (90% then 20%)
    gptscan.run_cli([], deep=False, show_all=True, use_gpt=False, rate_limit=60, output_format="csv", quiet=True, import_file=str(import_report), sort_by="threat", reverse_sort=False)
    out1 = capsys.readouterr().out
    lines1 = [line.split(",")[0] for line in out1.strip().splitlines() if line]
    assert lines1 == ["path", "high.py", "low.py"]

    # Reversed threat sorting: lowest first (20% then 90%)
    gptscan.run_cli([], deep=False, show_all=True, use_gpt=False, rate_limit=60, output_format="csv", quiet=True, import_file=str(import_report), sort_by="threat", reverse_sort=True)
    out2 = capsys.readouterr().out
    lines2 = [line.split(",")[0] for line in out2.strip().splitlines() if line]
    assert lines2 == ["path", "low.py", "high.py"]


def test_reverse_sort_default_sorting(tmp_path, capsys):
    import_report = tmp_path / "report.json"
    data = [
        {"path": "low.py", "line": "1", "own_conf": "20%", "admin_desc": "", "end-user_desc": "", "gpt_conf": "", "snippet": "low"},
        {"path": "high.py", "line": "1", "own_conf": "90%", "admin_desc": "", "end-user_desc": "", "gpt_conf": "", "snippet": "high"},
    ]
    import_report.write_text(json.dumps(data), encoding="utf-8")

    # Without sort_by, but reverse_sort=True, inverts default threat sorting
    gptscan.run_cli([], deep=False, show_all=True, use_gpt=False, rate_limit=60, output_format="csv", quiet=True, import_file=str(import_report), sort_by=None, reverse_sort=True)
    out = capsys.readouterr().out
    lines = [line.split(",")[0] for line in out.strip().splitlines() if line]
    assert lines == ["path", "low.py", "high.py"]


def test_cli_reverse_flags(monkeypatch, tmp_path):
    mock_run_cli = MagicMock(return_value=0)
    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)

    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')", encoding="utf-8")

    # Test short flag -r
    monkeypatch.setattr(sys, "argv", ["gptscan.py", str(test_file), "--cli", "-r"])
    gptscan.main()
    assert mock_run_cli.call_count == 1
    assert mock_run_cli.call_args.kwargs.get("reverse_sort") is True

    # Test long flag --reverse
    mock_run_cli.reset_mock()
    monkeypatch.setattr(sys, "argv", ["gptscan.py", str(test_file), "--cli", "--reverse"])
    gptscan.main()
    assert mock_run_cli.call_count == 1
    assert mock_run_cli.call_args.kwargs.get("reverse_sort") is True


def test_reverse_sort_live_scan(monkeypatch, capsys):
    events = [
        ('result', ("low.py", "20%", "", "", "", "low", "1")),
        ('result', ("high.py", "90%", "", "", "", "high", "1")),
        ('summary', (2, 100, 1.0))
    ]
    def mock_scan_files(*args, **kwargs):
        yield from events
    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    gptscan.run_cli(["dummy"], deep=False, show_all=True, use_gpt=False, rate_limit=60, output_format="csv", quiet=True, reverse_sort=True)
    out = capsys.readouterr().out
    lines = [line.split(",")[0] for line in out.strip().splitlines() if line]
    assert lines == ["path", "low.py", "high.py"]

    gptscan.run_cli(["dummy"], deep=False, show_all=True, use_gpt=False, rate_limit=60, output_format="json", quiet=True, reverse_sort=True)
    out_json = capsys.readouterr().out
    paths = [json.loads(line)["path"] for line in out_json.strip().splitlines() if line]
    assert paths == ["low.py", "high.py"]

