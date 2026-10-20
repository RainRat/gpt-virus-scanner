import os
import json
import sys
import pytest
from unittest.mock import patch, MagicMock
import gptscan


def test_load_preset_config_builtin():
    quick_cfg = gptscan.load_preset_config("quick")
    assert quick_cfg == {"threshold": 70, "max_depth": 2}

    ci_cfg = gptscan.load_preset_config("ci")
    assert ci_cfg == {"fail_threshold": 50, "threshold": 30, "deep": True, "all_files": True}

    strict_cfg = gptscan.load_preset_config("strict")
    assert strict_cfg == {"fail_threshold": 50, "threshold": 30, "deep": True, "all_files": True}

    audit_cfg = gptscan.load_preset_config("audit")
    assert audit_cfg == {"audit": True, "deep": True, "all_files": True}


def test_load_preset_config_json_file(tmp_path):
    preset_data = {
        "threshold": 40,
        "deep": True,
        "max_size": "5MB",
        "format": "json",
        "exclude": ["node_modules/*", "*.tmp"]
    }
    preset_file = tmp_path / "custom_preset.json"
    preset_file.write_text(json.dumps(preset_data), encoding="utf-8")

    loaded = gptscan.load_preset_config(str(preset_file))
    assert loaded == preset_data


def test_load_preset_config_yaml_file(tmp_path):
    yaml_content = """
threshold: 25
fail_threshold: 60
deep: true
extensions: "py,js,sh"
"""
    preset_file = tmp_path / "custom_preset.yaml"
    preset_file.write_text(yaml_content, encoding="utf-8")

    loaded = gptscan.load_preset_config(str(preset_file))
    assert loaded["threshold"] == 25
    assert loaded["fail_threshold"] == 60
    assert loaded["deep"] is True
    assert loaded["extensions"] == "py,js,sh"


def test_load_preset_config_errors():
    with pytest.raises(ValueError, match="cannot be empty"):
        gptscan.load_preset_config("")

    with pytest.raises(FileNotFoundError, match="not found"):
        gptscan.load_preset_config("non_existent_preset_file.json")

    with pytest.raises(ValueError, match="must contain a dictionary"):
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", pytest.importorskip("unittest.mock").mock_open(read_data="[1, 2, 3]")):
            gptscan.load_preset_config("invalid_top_level.json")


def test_load_preset_config_invalid_json(tmp_path):
    preset_file = tmp_path / "invalid.json"
    preset_file.write_text("{invalid json:", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse JSON preset file"):
        gptscan.load_preset_config(str(preset_file))


def test_load_preset_config_invalid_yaml(tmp_path):
    preset_file = tmp_path / "invalid.yaml"
    preset_file.write_text("threshold: [unclosed list", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse YAML preset file"):
        gptscan.load_preset_config(str(preset_file))


def test_load_preset_config_yaml_missing_pyyaml(tmp_path, monkeypatch):
    import builtins
    preset_file = tmp_path / "custom.yaml"
    preset_file.write_text("threshold: 50", encoding="utf-8")

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("No module named 'yaml'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    with pytest.raises(ValueError, match="PyYAML package is required"):
        gptscan.load_preset_config(str(preset_file))


def test_cli_preset_builtin_expansion(monkeypatch):
    test_args = ["gptscan.py", "--cli", "--preset", "quick", "--dry-run", "."]
    monkeypatch.setattr(sys, "argv", test_args)

    mock_run_cli = MagicMock(return_value=0)
    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)

    with pytest.raises(SystemExit) if False else patch.object(sys, "exit"):
        try:
            gptscan.main()
        except SystemExit:
            pass

    assert mock_run_cli.called
    call_kwargs = mock_run_cli.call_args[1]
    assert call_kwargs["max_depth"] == 2
    assert gptscan.Config.THRESHOLD == 70


def test_cli_preset_parameter_precedence(monkeypatch):
    test_args = ["gptscan.py", "--cli", "--preset", "quick", "--threshold", "85", "--max-depth", "5", "--dry-run", "."]
    monkeypatch.setattr(sys, "argv", test_args)

    mock_run_cli = MagicMock(return_value=0)
    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)

    try:
        gptscan.main()
    except SystemExit:
        pass

    assert mock_run_cli.called
    call_kwargs = mock_run_cli.call_args[1]
    assert call_kwargs["max_depth"] == 5
    assert gptscan.Config.THRESHOLD == 85
