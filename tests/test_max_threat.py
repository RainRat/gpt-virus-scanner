import os
import sys
import json
import pytest
import gptscan
from gptscan import Config, copy_cli_command, run_cli, main, load_preset_config


def test_config_max_threshold_default_and_persistence(tmp_path, monkeypatch):
    """Test Config.MAX_THRESHOLD default value, saving, and loading."""
    assert getattr(Config, 'MAX_THRESHOLD', 100) == 100

    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(Config, "SETTINGS_FILE", str(settings_file))
    monkeypatch.setattr(Config, "MAX_THRESHOLD", 80)

    Config.save_settings()
    assert os.path.exists(settings_file)

    with open(settings_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("max_threshold") == 80

    monkeypatch.setattr(Config, "MAX_THRESHOLD", 100)
    Config.load_settings()
    assert Config.MAX_THRESHOLD == 80


def test_copy_cli_command_max_threat(monkeypatch):
    """Test copy_cli_command appends --max-threat when MAX_THRESHOLD < 100."""
    clipboard_contents = []

    class DummyRoot:
        def clipboard_clear(self):
            clipboard_contents.clear()

        def clipboard_append(self, text):
            clipboard_contents.append(text)

        def update(self):
            pass

        def update_idletasks(self):
            pass

    monkeypatch.setattr(gptscan, "root", DummyRoot())
    monkeypatch.setattr(gptscan, "textbox", None)
    monkeypatch.setattr(Config, "last_path", "sample.py")
    monkeypatch.setattr(Config, "MAX_THRESHOLD", 75)

    copy_cli_command()
    assert len(clipboard_contents) == 1
    cmd = clipboard_contents[0]
    assert "--max-threat 75" in cmd or "--max-threat" in cmd


def test_preset_max_threat_parsing(tmp_path):
    """Test loading max_threat and max_threat_level from a preset file."""
    preset_json = tmp_path / "preset.json"
    preset_json.write_text(json.dumps({
        "min_threat": 20,
        "max_threat": 80
    }), encoding="utf-8")

    opts = load_preset_config(str(preset_json))
    assert opts.get("min_threat") == 20
    assert opts.get("max_threat") == 80


def test_run_cli_max_threat_filtering(tmp_path, monkeypatch):
    """Test run_cli filters out results above max_threat threshold."""
    # Dummy events with varying threat levels: "20%", "60%", "90%"
    events = [
        ('result', ('/path/file1.py', '20%', 'low threat', 'low threat', None, 'code1')),
        ('result', ('/path/file2.py', '60%', 'medium threat', 'medium threat', None, 'code2')),
        ('result', ('/path/file3.py', '90%', 'high threat', 'high threat', None, 'code3')),
    ]

    def mock_scan_files(*args, **kwargs):
        for ev in events:
            yield ev

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)
    monkeypatch.setattr(Config, "THRESHOLD", 50)

    # Unbounded max_threat (100) -> 60% and 90% are included (2 findings)
    threats_count_unbounded = run_cli(
        targets=str(tmp_path),
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        count_only=True,
        max_threat=100
    )
    assert threats_count_unbounded == 2

    # Bounded max_threat (80) -> only 60% is included (1 finding)
    threats_count_bounded = run_cli(
        targets=str(tmp_path),
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        count_only=True,
        max_threat=80
    )
    assert threats_count_bounded == 1


def test_cli_max_threat_validation(monkeypatch):
    """Test CLI argument validation for out-of-range --max-threat values."""
    monkeypatch.setattr(sys, "argv", ["gptscan.py", "--cli", "--max-threat", "150"])

    with pytest.raises(SystemExit):
        main()
