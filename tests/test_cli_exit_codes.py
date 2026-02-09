import gptscan
import pytest
import sys
import os

def test_run_cli_returns_threat_count(monkeypatch):
    def mock_scan_files(*args, **kwargs):
        # We need to yield at least one result
        # data format: (path, own_conf, admin, user, gpt_conf, snippet)
        yield ('result', ("f1.py", "90%", "", "", "", ""))
        yield ('result', ("f2.py", "40%", "", "", "", ""))
        yield ('summary', (2, 100, 1.0))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    # Test with default threshold (50)
    # Note: run_cli prints to stderr/stdout, we might want to suppress it
    monkeypatch.setattr(sys.stderr, "write", lambda x: None)

    count = gptscan.run_cli(["."], False, True, False, 60)
    assert count == 1

    # Test with custom threshold (30)
    count = gptscan.run_cli(["."], False, True, False, 60, fail_threshold=30)
    assert count == 2

    # Test with custom threshold (95)
    count = gptscan.run_cli(["."], False, True, False, 60, fail_threshold=95)
    assert count == 0

def test_main_exits_on_threshold(monkeypatch):
    # Mock run_cli to return 1 threat
    monkeypatch.setattr(gptscan, "run_cli", lambda *args, **kwargs: 1)

    # Mock sys.exit
    exit_codes = []
    monkeypatch.setattr(sys, "exit", lambda code: exit_codes.append(code))

    # Mock sys.argv
    monkeypatch.setattr(sys, "argv", ["gptscan.py", "--cli", "--fail-threshold", "50", "dummy_path"])

    # Mock other things needed for main to run
    monkeypatch.setattr(os.path, "exists", lambda x: True)
    monkeypatch.setattr(gptscan, "get_git_changed_files", lambda path=".": [])

    gptscan.main()

    assert 1 in exit_codes

def test_main_no_exit_if_no_threshold(monkeypatch):
    # Mock run_cli to return 1 threat
    monkeypatch.setattr(gptscan, "run_cli", lambda *args, **kwargs: 1)

    # Mock sys.exit
    exit_codes = []
    monkeypatch.setattr(sys, "exit", lambda code: exit_codes.append(code))

    # Mock sys.argv (no --fail-threshold)
    monkeypatch.setattr(sys, "argv", ["gptscan.py", "--cli", "dummy_path"])

    # Mock other things
    monkeypatch.setattr(os.path, "exists", lambda x: True)

    gptscan.main()

    assert 1 not in exit_codes

def test_env_var_apikey(monkeypatch):
    # Clear apikey if it was loaded
    monkeypatch.setattr(gptscan.Config, "apikey", "")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key-123")

    # Mock load_file to return empty string for apikey.txt
    original_load_file = gptscan.load_file
    def mock_load_file(filename, mode='single_line'):
        if filename == 'apikey.txt':
            return ""
        return original_load_file(filename, mode)
    monkeypatch.setattr(gptscan, "load_file", mock_load_file)

    gptscan.Config.initialize()
    assert gptscan.Config.apikey == "env-key-123"

def test_version_flag(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gptscan.py", "--version"])

    with pytest.raises(SystemExit) as e:
        gptscan.main()

    assert e.value.code == 0
    captured = capsys.readouterr()
    assert "1.1.0" in captured.out or "1.1.0" in captured.err
