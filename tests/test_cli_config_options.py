import sys
from unittest.mock import MagicMock
import pytest
import gptscan


@pytest.fixture(autouse=True)
def restore_config_state():
    orig_max_size = gptscan.Config.MAX_FILE_SIZE
    orig_scan_all = gptscan.Config.scan_all_files
    orig_exts = set(gptscan.Config.extensions_set) if gptscan.Config.extensions_set else set()
    orig_apikey = gptscan.Config.apikey
    orig_api_base = gptscan.Config.api_base
    orig_provider = gptscan.Config.provider
    orig_model = gptscan.Config.model_name

    yield

    gptscan.Config.MAX_FILE_SIZE = orig_max_size
    gptscan.Config.scan_all_files = orig_scan_all
    gptscan.Config.extensions_set = orig_exts
    gptscan.Config.apikey = orig_apikey
    gptscan.Config.api_base = orig_api_base
    gptscan.Config.provider = orig_provider
    gptscan.Config.model_name = orig_model


def test_cli_max_size_valid(monkeypatch, tmp_path):
    test_file = tmp_path / "sample.py"
    test_file.write_text("print('test')", encoding="utf-8")

    mock_run_cli = MagicMock(return_value=0)
    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)

    test_args = ["gptscan.py", str(test_file), "--cli", "--max-size", "20MB"]
    monkeypatch.setattr(sys, "argv", test_args)

    gptscan.main()

    assert gptscan.Config.MAX_FILE_SIZE == 20 * 1024 * 1024


def test_cli_max_size_invalid(monkeypatch, tmp_path):
    test_file = tmp_path / "sample.py"
    test_file.write_text("print('test')", encoding="utf-8")

    test_args = ["gptscan.py", str(test_file), "--cli", "--max-size", "invalid_size_format"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        gptscan.main()

    assert exc_info.value.code != 0


def test_cli_all_files(monkeypatch, tmp_path):
    test_file = tmp_path / "sample.py"
    test_file.write_text("print('test')", encoding="utf-8")

    mock_run_cli = MagicMock(return_value=0)
    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)

    test_args = ["gptscan.py", str(test_file), "--cli", "--all-files"]
    monkeypatch.setattr(sys, "argv", test_args)

    gptscan.main()

    assert gptscan.Config.scan_all_files is True


def test_cli_extensions(monkeypatch, tmp_path):
    test_file = tmp_path / "sample.py"
    test_file.write_text("print('test')", encoding="utf-8")

    mock_run_cli = MagicMock(return_value=0)
    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)

    test_args = ["gptscan.py", str(test_file), "--cli", "--extensions", "py,js,sh"]
    monkeypatch.setattr(sys, "argv", test_args)

    gptscan.main()

    assert ".py" in gptscan.Config.extensions_set
    assert ".js" in gptscan.Config.extensions_set
    assert ".sh" in gptscan.Config.extensions_set


def test_cli_api_key_and_base(monkeypatch, tmp_path):
    test_file = tmp_path / "sample.py"
    test_file.write_text("print('test')", encoding="utf-8")

    mock_run_cli = MagicMock(return_value=0)
    mock_save_apikey = MagicMock()

    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)
    monkeypatch.setattr(gptscan.Config, "save_apikey", mock_save_apikey)

    test_args = [
        "gptscan.py",
        str(test_file),
        "--cli",
        "--api-key", "test_key_12345",
        "--api-base", "http://localhost:8000/v1"
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    gptscan.main()

    assert gptscan.Config.apikey == "test_key_12345"
    assert gptscan.Config.api_base == "http://localhost:8000/v1"
    mock_save_apikey.assert_called_once()


def test_cli_provider_ollama_default_model(monkeypatch, tmp_path):
    test_file = tmp_path / "sample.py"
    test_file.write_text("print('test')", encoding="utf-8")

    mock_run_cli = MagicMock(return_value=0)
    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli)

    test_args = ["gptscan.py", str(test_file), "--cli", "--provider", "ollama"]
    monkeypatch.setattr(sys, "argv", test_args)

    gptscan.main()

    assert gptscan.Config.provider == "ollama"
    assert gptscan.Config.model_name == "llama3.2"
