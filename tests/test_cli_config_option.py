import json
import pytest
import gptscan


def test_load_config_file_valid(tmp_path):
    config_data = {
        "threshold": 75,
        "max_size": "5MB",
        "extensions": ["py", "sh"],
        "deep": True,
        "provider": "ollama",
        "model": "llama3.2"
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(config_data), encoding="utf-8")

    result = gptscan.load_config_file(str(cfg_file))
    assert result == config_data


def test_load_config_file_missing():
    with pytest.raises(ValueError, match="Configuration file not found"):
        gptscan.load_config_file("non_existent_config.json")


def test_load_config_file_invalid_json(tmp_path):
    cfg_file = tmp_path / "invalid.json"
    cfg_file.write_text("{ invalid json ", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON in configuration file"):
        gptscan.load_config_file(str(cfg_file))


def test_load_config_file_not_a_dict(tmp_path):
    cfg_file = tmp_path / "array.json"
    cfg_file.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="Configuration file content must be a JSON object"):
        gptscan.load_config_file(str(cfg_file))


def test_cli_config_option_integration(tmp_path, monkeypatch):
    test_file = tmp_path / "sample.py"
    test_file.write_text("print('hello')", encoding="utf-8")

    config_data = {
        "target": str(test_file),
        "threshold": 80,
        "deep": True,
        "all_files": True,
        "max_depth": 2,
        "format": "json"
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(config_data), encoding="utf-8")

    # Pass --config to CLI and verify config settings are respected
    monkeypatch.setattr("sys.argv", ["gptscan.py", "--cli", "--config", str(cfg_file)])

    captured = {}
    def mock_run_cli(scan_targets, deep, show_all, use_gpt, rate_limit, output_format=None, **kwargs):
        captured["scan_targets"] = scan_targets
        captured["deep"] = deep
        captured["output_format"] = output_format
        captured["max_depth"] = kwargs.get("max_depth")
        return 0

    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)
    monkeypatch.setattr(gptscan, "TK_AVAILABLE", False)

    try:
        gptscan.main()
    except SystemExit:
        pass

    assert captured["scan_targets"] == [str(test_file)]
    assert captured["deep"] is True
    assert captured["output_format"] == "json"
    assert captured["max_depth"] == 2
    assert gptscan.Config.THRESHOLD == 80
    assert gptscan.Config.scan_all_files is True


def test_cli_explicit_flags_override_config(tmp_path, monkeypatch):
    test_file = tmp_path / "sample.py"
    test_file.write_text("print('hello')", encoding="utf-8")

    config_data = {
        "target": "some_other_folder",
        "threshold": 90,
        "deep": False,
        "format": "json"
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(config_data), encoding="utf-8")

    # Explicit --threshold 30 and --deep should override config.json settings
    monkeypatch.setattr(
        "sys.argv",
        ["gptscan.py", str(test_file), "--cli", "--config", str(cfg_file), "--threshold", "30", "-d", "--format", "csv"]
    )

    captured = {}
    def mock_run_cli(scan_targets, deep, show_all, use_gpt, rate_limit, output_format=None, **kwargs):
        captured["scan_targets"] = scan_targets
        captured["deep"] = deep
        captured["output_format"] = output_format
        return 0

    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)
    monkeypatch.setattr(gptscan, "TK_AVAILABLE", False)

    try:
        gptscan.main()
    except SystemExit:
        pass

    assert captured["scan_targets"] == [str(test_file)]
    assert captured["deep"] is True
    assert captured["output_format"] == "csv"
    assert gptscan.Config.THRESHOLD == 30
