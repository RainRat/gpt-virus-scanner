import os
import pytest
from unittest.mock import MagicMock, patch
import gptscan
from gptscan import Config, manage_exclusions
import tkinter as tk

@pytest.fixture
def mock_gui_env(monkeypatch, tmp_path):
    # Setup temporary environment for file operations
    monkeypatch.chdir(tmp_path)
    Config.ignore_patterns = []

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
            if idx == "end":
                self.items.append(item)
            else:
                self.items.insert(idx, item)
        def delete(self, start, end):
            if end == "end":
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

    # Mock dialogs
    mock_simpledialog = MagicMock()
    monkeypatch.setattr(gptscan, 'simpledialog', mock_simpledialog)
    mock_filedialog = MagicMock()
    monkeypatch.setattr(gptscan, 'filedialog', mock_filedialog)
    mock_messagebox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_messagebox)

    # Mock _apply_filter
    mock_apply_filter = MagicMock()
    monkeypatch.setattr(gptscan, '_apply_filter', mock_apply_filter)

    return captured, mock_simpledialog, mock_filedialog, mock_messagebox, mock_toplevel

def test_manage_exclusions_add_pattern(mock_gui_env):
    captured, mock_sd, mock_fd, mock_mb, mock_top = mock_gui_env
    mock_sd.askstring.return_value = "*.log"

    manage_exclusions()
    add_btn, add_cmd = captured['buttons']['Add Pattern...']
    add_cmd()

    assert "*.log" in Config.ignore_patterns
    assert "*.log" in captured['listbox'].items
    assert gptscan._apply_filter.called

def test_manage_exclusions_add_folder(mock_gui_env, monkeypatch):
    captured, mock_sd, mock_fd, mock_mb, mock_top = mock_gui_env
    # Mock os.path.relpath to return a predictable relative path
    folder_path = "/tmp/some_folder"
    monkeypatch.setattr(os.path, 'relpath', lambda p, start: "some_folder")
    mock_fd.askdirectory.return_value = folder_path

    manage_exclusions()
    add_btn, add_cmd = captured['buttons']['Add Folder...']
    add_cmd()

    assert "some_folder" in Config.ignore_patterns
    assert "some_folder" in captured['listbox'].items
    assert gptscan._apply_filter.called

def test_manage_exclusions_remove_selected(mock_gui_env):
    captured, mock_sd, mock_fd, mock_mb, mock_top = mock_gui_env
    Config.ignore_patterns = ["p1", "p2", "p3"]
    mock_mb.askyesno.return_value = True

    manage_exclusions()
    lb = captured['listbox']
    # Selection is index-based. Listbox is populated in order of Config.ignore_patterns
    lb.selection = [1] # Select "p2"

    remove_btn, remove_cmd = captured['buttons']['Remove Selected']
    remove_cmd()

    assert "p2" not in Config.ignore_patterns
    assert "p1" in Config.ignore_patterns
    assert "p3" in Config.ignore_patterns
    assert gptscan._apply_filter.called

def test_manage_exclusions_add_pattern_cancel(mock_gui_env):
    captured, mock_sd, mock_fd, mock_mb, mock_top = mock_gui_env
    mock_sd.askstring.return_value = None

    manage_exclusions()
    add_cmd = captured['buttons']['Add Pattern...'][1]
    add_cmd()

    assert len(Config.ignore_patterns) == 0

def test_manage_exclusions_remove_no_selection(mock_gui_env):
    captured, mock_sd, mock_fd, mock_mb, mock_top = mock_gui_env
    Config.ignore_patterns = ["p1"]

    manage_exclusions()
    lb = captured['listbox']
    lb.selection = []

    remove_cmd = captured['buttons']['Remove Selected'][1]
    remove_cmd()

    assert len(Config.ignore_patterns) == 1
    mock_mb.askyesno.assert_not_called()

def test_manage_exclusions_remove_cancel(mock_gui_env):
    captured, mock_sd, mock_fd, mock_mb, mock_top = mock_gui_env
    Config.ignore_patterns = ["p1"]
    mock_mb.askyesno.return_value = False

    manage_exclusions()
    lb = captured['listbox']
    lb.selection = [0]

    remove_cmd = captured['buttons']['Remove Selected'][1]
    remove_cmd()

    assert len(Config.ignore_patterns) == 1
    mock_mb.askyesno.assert_called()

def test_manage_exclusions_add_pattern_error(mock_gui_env, monkeypatch):
    captured, mock_sd, mock_fd, mock_mb, mock_top = mock_gui_env
    mock_sd.askstring.return_value = "bad"
    monkeypatch.setattr(gptscan, "add_to_ignore_file", MagicMock(side_effect=Exception("Disk Error")))

    manage_exclusions()
    add_cmd = captured['buttons']['Add Pattern...'][1]
    add_cmd()

    mock_mb.showerror.assert_called_with("Error", "Could not update .gptscanignore: Disk Error", parent=mock_top)

def test_manage_exclusions_add_folder_error(mock_gui_env, monkeypatch):
    captured, mock_sd, mock_fd, mock_mb, mock_top = mock_gui_env
    mock_fd.askdirectory.return_value = "/some/dir"
    monkeypatch.setattr(os.path, 'relpath', MagicMock(side_effect=Exception("Path Error")))

    manage_exclusions()
    add_cmd = captured['buttons']['Add Folder...'][1]
    add_cmd()

    mock_mb.showerror.assert_called_with("Error", "Could not add folder: Path Error", parent=mock_top)

def test_manage_exclusions_remove_error(mock_gui_env, monkeypatch):
    captured, mock_sd, mock_fd, mock_mb, mock_top = mock_gui_env
    Config.ignore_patterns = ["p1"]
    mock_mb.askyesno.return_value = True
    monkeypatch.setattr(gptscan, "remove_from_ignore_file", MagicMock(side_effect=Exception("IO Error")))

    manage_exclusions()
    lb = captured['listbox']
    lb.selection = [0]

    remove_cmd = captured['buttons']['Remove Selected'][1]
    remove_cmd()

    mock_mb.showerror.assert_called_with("Error", "Could not update .gptscanignore: IO Error", parent=mock_top)
