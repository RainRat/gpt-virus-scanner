import sys
from pathlib import Path
import pytest
import gptscan

def test_min_threat_cli_option(monkeypatch, tmp_path, capsys):
    """Test that --min-threat sets Config.THRESHOLD and filters findings."""
    script1 = tmp_path / "safe.py"
    script1.write_text("print('hello world')\n", encoding="utf-8")

    test_args = [
        "gptscan.py",
        str(tmp_path),
        "--cli",
        "--min-threat", "70",
        "--count"
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    captured_thresholds = []

    def mock_run_cli(*args, **kwargs):
        captured_thresholds.append(gptscan.Config.THRESHOLD)
        return 0

    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)

    try:
        gptscan.main()
    except SystemExit:
        pass

    assert len(captured_thresholds) == 1
    assert captured_thresholds[0] == 70


def test_min_threat_level_alias(monkeypatch, tmp_path, capsys):
    """Test that --min-threat-level alias sets Config.THRESHOLD correctly."""
    script = tmp_path / "test.py"
    script.write_text("import os; os.system('echo test')\n", encoding="utf-8")

    test_args = [
        "gptscan.py",
        str(script),
        "--cli",
        "--min-threat-level", "85",
        "--count"
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    captured_thresholds = []

    def mock_run_cli(*args, **kwargs):
        captured_thresholds.append(gptscan.Config.THRESHOLD)
        return 0

    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)

    try:
        gptscan.main()
    except SystemExit:
        pass

    assert len(captured_thresholds) == 1
    assert captured_thresholds[0] == 85
