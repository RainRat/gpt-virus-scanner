import os
import sys
import pytest
from unittest.mock import MagicMock
import gptscan
from gptscan import parse_report_content, Config, _get_initial_dir

def test_parse_report_content_undecipherable():
    invalid_content = "completely random non-json plain text with no structure"
    with pytest.raises(ValueError, match="Could not determine report format."):
        parse_report_content(invalid_content)

def test_config_save_cache_error(monkeypatch, capsys):
    mock_open = MagicMock(side_effect=OSError("Disk full"))
    monkeypatch.setattr("builtins.open", mock_open)

    Config.save_cache()

    captured = capsys.readouterr()
    assert "Warning: Could not save AI cache:" in captured.err
    assert "Disk full" in captured.err

def test_config_load_cache_error(monkeypatch, capsys):
    monkeypatch.setattr(os.path, "exists", lambda x: True)

    mock_open = MagicMock(side_effect=OSError("Permission denied"))
    monkeypatch.setattr("builtins.open", mock_open)

    Config.load_cache()

    captured = capsys.readouterr()
    assert "Warning: Could not load AI cache:" in captured.err
    assert "Permission denied" in captured.err

def test_get_initial_dir_shlex_empty(monkeypatch):
    mock_textbox = MagicMock()
    mock_textbox.get.return_value = " # comment only "
    monkeypatch.setattr(gptscan, "textbox", mock_textbox)

    assert _get_initial_dir() is None

def test_is_supported_file_os_error(tmp_path):
    non_file_path = tmp_path / "test_directory"
    non_file_path.mkdir()

    assert Config.is_supported_file(non_file_path) is False
