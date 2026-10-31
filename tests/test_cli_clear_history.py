import sys
import pytest
import gptscan


def test_cli_clear_history_standalone_exits(monkeypatch, capsys):
    monkeypatch.setattr(gptscan.Config, "recent_paths", ["/path/one", "/path/two"])
    save_called = False

    def mock_save_settings():
        nonlocal save_called
        save_called = True

    monkeypatch.setattr(gptscan.Config, "save_settings", mock_save_settings)
    test_args = ["gptscan.py", "--cli", "--clear-history"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        gptscan.main()

    assert exc_info.value.code == 0
    assert gptscan.Config.recent_paths == []
    assert save_called is True
    captured = capsys.readouterr()
    assert "Path history cleared." in captured.err


def test_cli_clear_path_history_alias_standalone(monkeypatch, capsys):
    monkeypatch.setattr(gptscan.Config, "recent_paths", ["/prev/dir"])
    save_called = False

    def mock_save_settings():
        nonlocal save_called
        save_called = True

    monkeypatch.setattr(gptscan.Config, "save_settings", mock_save_settings)
    test_args = ["gptscan.py", "--cli", "--clear-path-history"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        gptscan.main()

    assert exc_info.value.code == 0
    assert gptscan.Config.recent_paths == []
    assert save_called is True
    captured = capsys.readouterr()
    assert "Path history cleared." in captured.err


def test_cli_clear_history_quiet_mode(monkeypatch, capsys):
    monkeypatch.setattr(gptscan.Config, "recent_paths", ["/some/path"])
    save_called = False

    def mock_save_settings():
        nonlocal save_called
        save_called = True

    monkeypatch.setattr(gptscan.Config, "save_settings", mock_save_settings)
    test_args = ["gptscan.py", "--cli", "--clear-history", "--quiet"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        gptscan.main()

    assert exc_info.value.code == 0
    assert gptscan.Config.recent_paths == []
    assert save_called is True
    captured = capsys.readouterr()
    assert captured.err == ""


def test_cli_clear_history_with_target_continues_execution(monkeypatch, capsys):
    monkeypatch.setattr(gptscan.Config, "recent_paths", ["/old/target"])
    save_called = False

    def mock_save_settings():
        nonlocal save_called
        save_called = True

    called_run_cli = False

    def mock_run_cli_func(*args, **kwargs):
        nonlocal called_run_cli
        called_run_cli = True
        return 0

    monkeypatch.setattr(gptscan, "run_cli", mock_run_cli_func)
    monkeypatch.setattr(gptscan.Config, "save_settings", mock_save_settings)
    test_args = ["gptscan.py", "--cli", "--clear-history", "."]
    monkeypatch.setattr(sys, "argv", test_args)

    gptscan.main()

    assert gptscan.Config.recent_paths == []
    assert save_called is True
    assert called_run_cli is True
    captured = capsys.readouterr()
    assert "Path history cleared." in captured.err
