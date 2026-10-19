import json
import pytest
import gptscan
from gptscan import Config


def test_cli_extensions_file_valid(tmp_path, monkeypatch):
    """Test that `--extensions-file` correctly parses extensions from a text file."""
    ext_file = tmp_path / "custom_exts.txt"
    ext_file.write_text("py\n# comment line\n.js, ts\n  sh  \n", encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["gptscan.py", "--cli", "--dry-run", "--extensions-file", str(ext_file)])

    # Save original extensions set
    orig_exts = set(Config.extensions_set)
    try:
        gptscan.main()
        assert Config.extensions_set == {".py", ".js", ".ts", ".sh"}
    finally:
        Config.extensions_set = orig_exts


def test_cli_extensions_file_and_extensions_combined(tmp_path, monkeypatch):
    """Test combining `--extensions` and `--extensions-file`."""
    ext_file = tmp_path / "exts.txt"
    ext_file.write_text("py\njs\n", encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["gptscan.py", "--cli", "--dry-run", "--extensions-file", str(ext_file), "--extensions", "rb,go"])

    orig_exts = set(Config.extensions_set)
    try:
        gptscan.main()
        assert Config.extensions_set == {".py", ".js", ".rb", ".go"}
    finally:
        Config.extensions_set = orig_exts


def test_cli_extensions_file_not_found(tmp_path, monkeypatch):
    """Test error handling when `--extensions-file` points to a non-existent file."""
    non_existent = tmp_path / "does_not_exist.txt"

    monkeypatch.setattr("sys.argv", ["gptscan.py", "--cli", "--dry-run", "--extensions-file", str(non_existent)])

    with pytest.raises(SystemExit):
        gptscan.main()


def test_cli_extensions_file_via_config(tmp_path, monkeypatch):
    """Test loading `extensions_file` from a config JSON file."""
    ext_file = tmp_path / "custom_exts.txt"
    ext_file.write_text("php\npl\n", encoding="utf-8")

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"extensions_file": str(ext_file)}), encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["gptscan.py", "--cli", "--dry-run", "--config", str(config_file)])

    orig_exts = set(Config.extensions_set)
    try:
        gptscan.main()
        assert Config.extensions_set == {".php", ".pl"}
    finally:
        Config.extensions_set = orig_exts
