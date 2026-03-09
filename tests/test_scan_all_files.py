import pytest
from pathlib import Path
from gptscan import Config, scan_files

def test_is_supported_file_with_scan_all_files(monkeypatch):
    monkeypatch.setattr(Config, "scan_all_files", True)
    # .txt is normally not supported without shebang
    assert Config.is_supported_file(Path("test.txt")) is True

    monkeypatch.setattr(Config, "scan_all_files", False)
    # Ensure .txt is NOT in extensions_set for this test
    monkeypatch.setattr(Config, "extensions_set", {".py", ".js"})
    assert Config.is_supported_file(Path("test.txt")) is False
    # .py is always supported
    assert Config.is_supported_file(Path("test.py")) is True

def test_scan_files_respects_scan_all_files(tmp_path, monkeypatch):
    # Create a non-supported file
    f = tmp_path / "test.txt"
    f.write_text("plain text file")

    # 1. scan_all_files = False (Default)
    monkeypatch.setattr(Config, "scan_all_files", False)
    monkeypatch.setattr(Config, "extensions_set", {".py"}) # Ensure .txt is not in there

    results = list(scan_files(str(tmp_path), deep_scan=False, show_all=True, use_gpt=False, dry_run=True))

    # Should be empty or at least not contain test.txt (except for progress/summary events)
    result_paths = [data[0] for event, data in results if event == 'result']
    assert str(f) not in result_paths

    # 2. scan_all_files = True
    monkeypatch.setattr(Config, "scan_all_files", True)
    results = list(scan_files(str(tmp_path), deep_scan=False, show_all=True, use_gpt=False, dry_run=True))

    result_paths = [data[0] for event, data in results if event == 'result']
    assert str(f) in result_paths
