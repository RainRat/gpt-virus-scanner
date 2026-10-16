import json
import os
import sys
import pytest
import gptscan


def test_config_max_threshold_defaults_and_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(gptscan.Config, "SETTINGS_FILE", str(tmp_path / ".gptscan_settings.json"))
    monkeypatch.setattr(gptscan.Config, "MAX_THRESHOLD", 100)
    monkeypatch.setattr(gptscan.Config, "THRESHOLD", 50)

    # Save default settings
    gptscan.Config.save_settings()
    assert os.path.exists(gptscan.Config.SETTINGS_FILE)

    with open(gptscan.Config.SETTINGS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("max_threshold") == 100

    # Modify settings file with custom max_threshold
    data["max_threshold"] = 75
    with open(gptscan.Config.SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)

    gptscan.Config.load_settings()
    assert gptscan.Config.MAX_THRESHOLD == 75


def test_cli_max_threat_parsing_and_config_update(monkeypatch):
    monkeypatch.setattr(gptscan.Config, "MAX_THRESHOLD", 100)
    monkeypatch.setattr(gptscan.Config, "THRESHOLD", 50)

    called_run_cli = False

    def mock_run_cli(*args, **kwargs):
        nonlocal called_run_cli
        called_run_cli = True
        assert kwargs.get("max_threat") == 80
        return 0

    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)
    test_args = ["gptscan.py", ".", "--cli", "--max-threat", "80"]
    monkeypatch.setattr(sys, "argv", test_args)

    gptscan.main()
    assert called_run_cli
    assert gptscan.Config.MAX_THRESHOLD == 80


def test_cli_max_threat_invalid_value(monkeypatch):
    test_args = ["gptscan.py", ".", "--cli", "--max-threat", "150"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        gptscan.main()
    assert exc_info.value.code != 0


def test_cli_min_threat_greater_than_max_threat(monkeypatch):
    test_args = ["gptscan.py", ".", "--cli", "--min-threat", "80", "--max-threat", "50"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        gptscan.main()
    assert exc_info.value.code != 0


def test_run_cli_max_threat_filtering(capsys, monkeypatch):
    monkeypatch.setattr(gptscan.Config, "THRESHOLD", 30)
    monkeypatch.setattr(gptscan.Config, "MAX_THRESHOLD", 100)

    # Mock scan_files to yield findings with different threat scores
    sample_events = [
        ('result', ('file1.py', '20%', 'Admin note', 'User note', '20%', 'print("low")')),
        ('result', ('file2.py', '50%', 'Admin note', 'User note', '50%', 'import os; os.system("cmd")')),
        ('result', ('file3.py', '90%', 'Admin note', 'User note', '90%', 'eval("harmful")')),
        ('summary', (3, 1024, 0.5)),
    ]

    def mock_scan_files(*args, **kwargs):
        for ev in sample_events:
            yield ev

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    # Run CLI with max_threat = 70
    threats = gptscan.run_cli(
        targets=["."],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        quiet=True,
        max_threat=70,
    )

    # Only file2.py (threat 50) falls in range [30, 70]
    assert threats == 1
    captured = capsys.readouterr()
    assert "file2.py" in captured.out
    assert "file3.py" not in captured.out
    assert "file1.py" not in captured.out


def test_run_cli_paths_only_with_max_threat(capsys, monkeypatch):
    monkeypatch.setattr(gptscan.Config, "THRESHOLD", 20)

    sample_events = [
        ('result', ('file_low.py', '25%', 'Admin', 'User', '25%', 'code')),
        ('result', ('file_med.py', '60%', 'Admin', 'User', '60%', 'code')),
        ('result', ('file_high.py', '95%', 'Admin', 'User', '95%', 'code')),
        ('summary', (3, 500, 0.2)),
    ]

    monkeypatch.setattr(gptscan, "scan_files", lambda *args, **kwargs: (ev for ev in sample_events))

    gptscan.run_cli(
        targets=["."],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        quiet=True,
        paths_only=True,
        max_threat=70,
    )

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
    assert "file_low.py" in lines
    assert "file_med.py" in lines
    assert "file_high.py" not in lines


def test_copy_cli_command_with_max_threat(monkeypatch):
    copied_cmd = ""

    class MockRoot:
        def clipboard_clear(self):
            pass

        def clipboard_append(self, text):
            nonlocal copied_cmd
            copied_cmd = text

    monkeypatch.setattr(gptscan.Config, "last_path", "/tmp/scan_target")
    monkeypatch.setattr(gptscan.Config, "MAX_THRESHOLD", 75)
    monkeypatch.setattr(gptscan.Config, "THRESHOLD", 50)
    monkeypatch.setattr(gptscan, "root", MockRoot())
    monkeypatch.setattr(gptscan, "textbox", None)
    monkeypatch.setattr(gptscan, "deep_var", None)
    monkeypatch.setattr(gptscan, "git_var", None)
    monkeypatch.setattr(gptscan, "dry_var", None)
    monkeypatch.setattr(gptscan, "all_var", None)
    monkeypatch.setattr(gptscan, "scan_all_var", None)
    monkeypatch.setattr(gptscan, "gpt_var", None)
    monkeypatch.setattr(gptscan, "update_status", lambda msg: None)

    gptscan.copy_cli_command()
    assert "--max-threat 75" in copied_cmd
