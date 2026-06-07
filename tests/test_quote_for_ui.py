import sys
import shlex
import pytest
from gptscan import _quote_for_ui

def test_quote_for_ui_posix_no_spaces(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    path = "/home/user/file.py"
    # shlex.quote behavior might vary slightly by python version,
    # but for simple paths it usually returns it as is or single quoted.
    assert _quote_for_ui(path) == shlex.quote(path)

def test_quote_for_ui_posix_with_spaces(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    path = "/home/user/my file.py"
    assert _quote_for_ui(path) == shlex.quote(path)

def test_quote_for_ui_windows_no_special_chars(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    path = "C:\\Users\\file.py"
    assert _quote_for_ui(path) == path

def test_quote_for_ui_windows_with_spaces(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    path = "C:\\Users\\my file.py"
    assert _quote_for_ui(path) == '"C:\\Users\\my file.py"'

def test_quote_for_ui_windows_with_special_chars(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    special_chars = '%&^|<>'
    for char in special_chars:
        path = f"C:\\temp\\file{char}.py"
        assert _quote_for_ui(path) == f'"{path}"'

def test_quote_for_ui_windows_escapes_double_quotes(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    path = 'C:\\temp\\"file".py'
    assert _quote_for_ui(path) == '"C:\\temp\\""file"".py"'
