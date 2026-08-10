import pytest
from unittest.mock import MagicMock, patch
import gptscan
import tkinter as tk
import sys

@pytest.fixture
def mock_shortcuts_env(monkeypatch):
    mock_root = MagicMock()
    monkeypatch.setattr(gptscan, 'root', mock_root)

    # Mock Toplevel
    mock_toplevel = MagicMock()
    mock_toplevel.winfo_children.return_value = []
    monkeypatch.setattr(gptscan.tk, 'Toplevel', MagicMock(return_value=mock_toplevel))

    # Mock Canvas
    mock_canvas = MagicMock()
    monkeypatch.setattr(gptscan.tk, 'Canvas', MagicMock(return_value=mock_canvas))

    # Mock Notebook
    mock_notebook = MagicMock()
    monkeypatch.setattr(gptscan.ttk, 'Notebook', MagicMock(return_value=mock_notebook))

    # Mock Scrollbar
    mock_scrollbar = MagicMock()
    monkeypatch.setattr(gptscan.ttk, 'Scrollbar', MagicMock(return_value=mock_scrollbar))

    # Mock Label
    class MockLabel:
        def __init__(self, master=None, text="", **kwargs):
            self.master = master
            self.text = text
            self.kwargs = kwargs
        def pack(self, **kwargs): pass
        def grid(self, **kwargs): pass
        def winfo_children(self): return []

    monkeypatch.setattr(gptscan.ttk, 'Label', MockLabel)

    # Mock Button
    mock_button = MagicMock()
    monkeypatch.setattr(gptscan.ttk, 'Button', MagicMock(return_value=mock_button))

    # Mock Separator
    mock_separator = MagicMock()
    monkeypatch.setattr(gptscan.ttk, 'Separator', MagicMock(return_value=mock_separator))

    captured = {
        'toplevel': mock_toplevel,
        'canvas': mock_canvas,
        'notebook': mock_notebook,
        'labels': []
    }

    # Intercept label creation
    def mock_label_intercept(master=None, text="", **kwargs):
        lbl = MockLabel(master, text, **kwargs)
        captured['labels'].append(lbl)
        return lbl
    monkeypatch.setattr(gptscan.ttk, 'Label', mock_label_intercept)

    return captured

def test_show_keyboard_shortcuts_opens_window(mock_shortcuts_env):
    captured = mock_shortcuts_env
    gptscan.show_keyboard_shortcuts()

    # Verify Toplevel was opened and setup
    gptscan.tk.Toplevel.assert_called_once_with(gptscan.root)
    captured['toplevel'].title.assert_called_with("Keyboard Shortcuts")
    captured['toplevel'].transient.assert_called_with(gptscan.root)
    captured['toplevel'].grab_set.assert_called_once()
    captured['toplevel'].focus_set.assert_called_once()

    # Verify Notebook and Canvas were instantiated
    gptscan.ttk.Notebook.assert_called_once()
    assert gptscan.tk.Canvas.call_count == 3  # 3 tabs, 1 canvas per tab

def test_show_keyboard_shortcuts_bindings(mock_shortcuts_env):
    captured = mock_shortcuts_env
    bindings = {}
    captured['toplevel'].bind.side_effect = lambda event, func: bindings.update({event: func})

    gptscan.show_keyboard_shortcuts()

    assert '<Escape>' in bindings
    assert '<Return>' in bindings

    # Trigger Esc binding and check it destroys Toplevel
    bindings['<Escape>'](None)
    captured['toplevel'].destroy.assert_called_once()

def test_show_keyboard_shortcuts_content_and_tabs(mock_shortcuts_env):
    captured = mock_shortcuts_env
    gptscan.show_keyboard_shortcuts()

    # Check that the 3 tabs were added
    assert captured['notebook'].add.call_count == 3
    added_texts = [call[1].get('text') for call in captured['notebook'].add.call_args_list]
    assert "General / Navigation" in added_texts
    assert "Results List" in added_texts
    assert "Details Window" in added_texts

    # Check some labels to verify shortcut key rendering
    label_texts = [lbl.text for lbl in captured['labels']]
    assert "Keyboard Shortcuts Reference" in label_texts
    assert "Enter" in label_texts
    assert "Esc" in label_texts
    assert "Space / Enter" in label_texts

def test_show_keyboard_shortcuts_darwin_modifier(mock_shortcuts_env, monkeypatch):
    captured = mock_shortcuts_env
    monkeypatch.setattr(sys, 'platform', 'darwin')

    gptscan.show_keyboard_shortcuts()

    label_texts = [lbl.text for lbl in captured['labels']]
    # Verify Cmd+F is used on Mac
    assert "Cmd+F" in label_texts
    assert "Cmd+O" in label_texts
    # Verify Ctrl+F is NOT used on Mac
    assert "Ctrl+F" not in label_texts

def test_show_keyboard_shortcuts_linux_modifier(mock_shortcuts_env, monkeypatch):
    captured = mock_shortcuts_env
    monkeypatch.setattr(sys, 'platform', 'linux')

    gptscan.show_keyboard_shortcuts()

    label_texts = [lbl.text for lbl in captured['labels']]
    # Verify Ctrl+F is used on Linux
    assert "Ctrl+F" in label_texts
    assert "Ctrl+O" in label_texts
    # Verify Cmd+F is NOT used on Linux
    assert "Cmd+F" not in label_texts

def test_show_keyboard_shortcuts_no_root(monkeypatch):
    monkeypatch.setattr(gptscan, 'root', None)
    with patch('tkinter.Toplevel') as mock_toplevel:
        gptscan.show_keyboard_shortcuts()
        mock_toplevel.assert_not_called()
