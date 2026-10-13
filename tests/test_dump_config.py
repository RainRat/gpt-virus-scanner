import json
import sys
import pytest
from unittest.mock import patch
import gptscan


@pytest.fixture(autouse=True)
def reset_config_state():
    orig_threshold = gptscan.Config.THRESHOLD
    orig_scan_all = gptscan.Config.scan_all_files
    orig_provider = gptscan.Config.provider
    orig_model = gptscan.Config.model_name
    orig_exts = set(gptscan.Config.extensions_set)
    yield
    gptscan.Config.THRESHOLD = orig_threshold
    gptscan.Config.scan_all_files = orig_scan_all
    gptscan.Config.provider = orig_provider
    gptscan.Config.model_name = orig_model
    gptscan.Config.extensions_set = orig_exts


def test_generate_config_export_basic():
    cfg = gptscan.generate_config_export(
        provider="openai",
        model="gpt-4o",
        threshold=75,
        deep=True,
        exclude=["*.log", "temp/*"],
        include=["src/*"]
    )
    assert cfg["provider"] == "openai"
    assert cfg["model"] == "gpt-4o"
    assert cfg["threshold"] == 75
    assert cfg["deep"] is True
    assert cfg["exclude"] == ["*.log", "temp/*"]
    assert cfg["include"] == ["src/*"]


def test_export_config_to_string_json():
    data = {"provider": "ollama", "model": "llama3.2", "threshold": 50}
    json_str = gptscan.export_config_to_string(data, format_type="json")
    parsed = json.loads(json_str)
    assert parsed["provider"] == "ollama"
    assert parsed["model"] == "llama3.2"
    assert parsed["threshold"] == 50


def test_export_config_to_string_yaml():
    data = {"provider": "openrouter", "threshold": 60}
    try:
        yaml_str = gptscan.export_config_to_string(data, format_type="yaml")
        assert "provider: openrouter" in yaml_str
        assert "threshold: 60" in yaml_str
    except ValueError as e:
        assert "PyYAML module is required" in str(e)


def test_export_config_to_string_yaml_missing_module(monkeypatch):
    import builtins
    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("No module named yaml")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ValueError, match="PyYAML module is required"):
        gptscan.export_config_to_string({"a": 1}, format_type="yaml")


def test_cli_dump_config_stdout(capsys, monkeypatch):
    test_args = ["gptscan.py", "--dump-config"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        gptscan.main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    cfg = json.loads(captured.out)
    assert "provider" in cfg
    assert "threshold" in cfg
    assert "extensions" in cfg


def test_cli_dump_config_file(tmp_path, monkeypatch):
    output_file = tmp_path / "config_dump.json"
    test_args = ["gptscan.py", "--dump-config", str(output_file), "--threshold", "85", "-e", "*.tmp"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        gptscan.main()

    assert exc_info.value.code == 0
    assert output_file.exists()

    with open(output_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["threshold"] == 85
    assert "*.tmp" in cfg["exclude"]


def test_cli_dump_config_preset(capsys, monkeypatch):
    test_args = ["gptscan.py", "-P", "audit", "--dump-config"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        gptscan.main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    cfg = json.loads(captured.out)
    assert cfg["deep"] is True
    assert cfg["all_files"] is True


def test_cli_dump_config_yaml(tmp_path, monkeypatch):
    output_file = tmp_path / "config_dump.yaml"
    test_args = ["gptscan.py", "--dump-config", str(output_file), "-P", "quick"]
    monkeypatch.setattr(sys, "argv", test_args)

    try:
        with pytest.raises(SystemExit) as exc_info:
            gptscan.main()
        assert exc_info.value.code == 0
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "threshold:" in content
    except SystemExit as exc_info:
        assert exc_info.code in (0, 2)


def test_config_dump_reimport_symmetry(tmp_path, monkeypatch):
    dump_file = tmp_path / "exported_config.json"
    dump_args = ["gptscan.py", "--threshold", "92", "-e", "build/*", "--dump-config", str(dump_file)]
    monkeypatch.setattr(sys, "argv", dump_args)

    with pytest.raises(SystemExit) as exc_info:
        gptscan.main()
    assert exc_info.value.code == 0

    reimported = gptscan.load_config_file(str(dump_file))
    assert reimported["threshold"] == 92
    assert "build/*" in reimported["exclude"]
