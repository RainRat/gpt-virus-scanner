import sys
import pytest
import gptscan


def test_cli_clear_cache_standalone_exits(monkeypatch, capsys):
    monkeypatch.setattr(gptscan.Config, "gpt_cache", {"item1": "data1"})
    save_called = False

    def mock_save_cache():
        nonlocal save_called
        save_called = True

    monkeypatch.setattr(gptscan.Config, "save_cache", mock_save_cache)
    test_args = ["gptscan.py", "--cli", "--clear-cache"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        gptscan.main()

    assert exc_info.value.code == 0
    assert gptscan.Config.gpt_cache == {}
    assert save_called is True
    captured = capsys.readouterr()
    assert "AI Analysis cache cleared." in captured.err


def test_cli_clear_cache_quiet_mode(monkeypatch, capsys):
    monkeypatch.setattr(gptscan.Config, "gpt_cache", {"key": "val"})
    save_called = False

    def mock_save_cache():
        nonlocal save_called
        save_called = True

    monkeypatch.setattr(gptscan.Config, "save_cache", mock_save_cache)
    test_args = ["gptscan.py", "--cli", "--clear-cache", "--quiet"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        gptscan.main()

    assert exc_info.value.code == 0
    assert gptscan.Config.gpt_cache == {}
    assert save_called is True
    captured = capsys.readouterr()
    assert captured.err == ""


def test_cli_clear_cache_with_target_continues_execution(monkeypatch, capsys):
    monkeypatch.setattr(gptscan.Config, "gpt_cache", {"cached": "entry"})
    save_called = False

    def mock_save_cache():
        nonlocal save_called
        save_called = True

    called_run_cli = False

    def mock_run_cli_func(*args, **kwargs):
        nonlocal called_run_cli
        called_run_cli = True
        return 0

    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli_func)
    monkeypatch.setattr(gptscan.Config, "save_cache", mock_save_cache)
    test_args = ["gptscan.py", "--cli", "--clear-cache", "."]
    monkeypatch.setattr(sys, "argv", test_args)

    gptscan.main()

    assert gptscan.Config.gpt_cache == {}
    assert save_called is True
    assert called_run_cli is True
    captured = capsys.readouterr()
    assert "AI Analysis cache cleared." in captured.err
