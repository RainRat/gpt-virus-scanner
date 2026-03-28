import os
import pytest
from unittest.mock import MagicMock, patch
import gptscan
from gptscan import Config, manage_extensions
import tkinter as tk

@pytest.fixture
def mock_gui_env(monkeypatch):
    mock_root = MagicMock()
    monkeypatch.setattr(gptscan, 'root', mock_root)

    # Mock Toplevel
    mock_toplevel = MagicMock()
    monkeypatch.setattr(gptscan.tk, 'Toplevel', MagicMock(return_value=mock_toplevel))

    # Mock Listbox
    class MockListbox:
        def __init__(self, *args, **kwargs):
            self.items = []
            self.selection = []
        def insert(self, idx, item):
            if idx == gptscan.tk.END:
                self.items.append(item)
            else:
                self.items.insert(idx, item)
        def delete(self, start, end):
            if end == gptscan.tk.END:
                self.items = self.items[:start]
            else:
                del self.items[start:end+1]
        def get(self, idx):
            return self.items[idx]
        def curselection(self):
            return self.selection
        def pack(self, **kwargs): pass
        def config(self, **kwargs): pass
        def yview(self, *args): pass

    monkeypatch.setattr(gptscan.tk, 'Listbox', MockListbox)

    # Capture components
    captured = {
        'buttons': {},
        'listbox': None
    }

    def mock_button_init(master, **kwargs):
        btn = MagicMock()
        text = kwargs.get('text', '')
        if text:
            captured['buttons'][text] = (btn, kwargs.get('command'))
        return btn
    monkeypatch.setattr(gptscan.ttk, 'Button', mock_button_init)

    original_listbox = gptscan.tk.Listbox
    def mock_listbox_init(*args, **kwargs):
        lb = original_listbox(*args, **kwargs)
        captured['listbox'] = lb
        return lb
    monkeypatch.setattr(gptscan.tk, 'Listbox', mock_listbox_init)

    # Mock simpledialog and messagebox
    mock_simpledialog = MagicMock()
    monkeypatch.setattr(gptscan, 'simpledialog', mock_simpledialog)
    mock_messagebox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_messagebox)

    return captured, mock_simpledialog, mock_messagebox, mock_toplevel

def test_save_extensions(tmp_path):
    # Change to temporary directory to avoid overwriting real extensions.txt
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        Config.extensions_set = {".test1", ".test2"}
        Config.save_extensions()

        assert os.path.exists("extensions.txt")
        with open("extensions.txt", "r") as f:
            lines = f.read().splitlines()

        assert sorted(lines) == [".test1", ".test2"]
    finally:
        os.chdir(original_cwd)

def test_set_extensions_normalization():
    Config.set_extensions(["py", ".js", "  BAT  "])
    assert ".py" in Config.extensions_set
    assert ".js" in Config.extensions_set
    assert ".bat" in Config.extensions_set
    assert len(Config.extensions_set) == 3

def test_manage_extensions_init(mock_gui_env):
    captured, mock_sd, mock_mb, mock_top = mock_gui_env
    Config.extensions_set = {".py", ".js"}

    manage_extensions()

    assert mock_top.title.call_args[0][0] == "Manage Extensions"
    assert captured['listbox'].items == [".js", ".py"]

def test_manage_extensions_add(mock_gui_env, monkeypatch):
    captured, mock_sd, mock_mb, mock_top = mock_gui_env
    Config.extensions_set = {".py"}
    mock_sd.askstring.return_value = "rb"

    # Mock save_extensions to avoid file IO during this test if possible,
    # but gptscan already uses it. We can just let it run or mock it.
    monkeypatch.setattr(Config, "save_extensions", MagicMock())

    manage_extensions()
    add_btn, add_cmd = captured['buttons']['Add...']
    add_cmd()

    assert ".rb" in Config.extensions_set
    assert ".rb" in captured['listbox'].items
    Config.save_extensions.assert_called()

def test_manage_extensions_remove(mock_gui_env, monkeypatch):
    captured, mock_sd, mock_mb, mock_top = mock_gui_env
    Config.extensions_set = {".py", ".js"}
    mock_mb.askyesno.return_value = True
    monkeypatch.setattr(Config, "save_extensions", MagicMock())

    manage_extensions()
    lb = captured['listbox']
    # Select .js (index 0 because they are sorted)
    lb.selection = [0]

    remove_btn, remove_cmd = captured['buttons']['Remove Selected']
    remove_cmd()

    assert ".js" not in Config.extensions_set
    assert ".py" in Config.extensions_set
    mock_mb.askyesno.assert_called()
    Config.save_extensions.assert_called()

def test_manage_extensions_reset(mock_gui_env, monkeypatch):
    captured, mock_sd, mock_mb, mock_top = mock_gui_env
    Config.extensions_set = {".custom"}
    mock_mb.askyesno.return_value = True
    monkeypatch.setattr(Config, "save_extensions", MagicMock())

    manage_extensions()
    reset_btn, reset_cmd = captured['buttons']['Reset to Defaults']
    reset_cmd()

    assert ".py" in Config.extensions_set
    assert ".custom" not in Config.extensions_set
    Config.save_extensions.assert_called()
