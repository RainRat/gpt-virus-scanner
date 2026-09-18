import pytest
import tkinter as tk
from unittest.mock import MagicMock
import gptscan


def test_keypad_enter_bindings_main_gui(monkeypatch):
    """Test that <KP_Enter> and its modifiers are properly bound on main GUI widgets."""
    mock_textbox = MagicMock()
    textbox_bindings = {}
    mock_textbox.bind.side_effect = lambda event, func: textbox_bindings.update({event: func})

    mock_filter_entry = MagicMock()
    filter_bindings = {}
    mock_filter_entry.bind.side_effect = lambda event, func: filter_bindings.update({event: func})

    mock_tree = MagicMock()
    tree_bindings = {}
    mock_tree.bind.side_effect = lambda event, func: tree_bindings.update({event: func})

    mock_root = MagicMock()
    root_bindings = {}
    mock_root.bind.side_effect = lambda event, func: root_bindings.update({event: func})

    mock_button_click = MagicMock()
    mock_on_filter_return = MagicMock()
    mock_on_root_return = MagicMock()

    monkeypatch.setattr(gptscan, 'button_click', mock_button_click)
    monkeypatch.setattr(gptscan, 'on_filter_return', mock_on_filter_return)
    monkeypatch.setattr(gptscan, 'on_root_return', mock_on_root_return)

    monkeypatch.setattr(gptscan, 'textbox', mock_textbox)
    monkeypatch.setattr(gptscan, 'filter_entry', mock_filter_entry)
    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    monkeypatch.setattr(gptscan, 'root', mock_root)

    # Instantiate mock GUI to capture bindings
    with monkeypatch.context() as m:
        m.setattr(gptscan.tk, 'Tk', lambda: mock_root)
        m.setattr(gptscan.ttk, 'Combobox', lambda *a, **kw: mock_textbox)
        m.setattr(gptscan.ttk, 'Entry', lambda *a, **kw: mock_filter_entry)
        m.setattr(gptscan.ttk, 'Treeview', lambda *a, **kw: mock_tree)

        gptscan.create_gui()

    # Verify <KP_Enter> on textbox
    assert '<KP_Enter>' in textbox_bindings
    textbox_bindings['<KP_Enter>'](None)
    mock_button_click.assert_called_once()

    # Verify <KP_Enter> on filter_entry
    assert '<KP_Enter>' in filter_bindings
    filter_bindings['<KP_Enter>'](None)
    mock_on_filter_return.assert_called_once()

    # Verify <KP_Enter> and modifiers on tree
    assert '<KP_Enter>' in tree_bindings
    assert '<Shift-KP_Enter>' in tree_bindings
    assert '<Control-KP_Enter>' in tree_bindings
    assert '<Command-KP_Enter>' in tree_bindings

    assert tree_bindings['<KP_Enter>'] == gptscan.view_details
    assert tree_bindings['<Shift-KP_Enter>'] == gptscan.open_file
    assert tree_bindings['<Control-KP_Enter>'] == gptscan.show_in_folder
    assert tree_bindings['<Command-KP_Enter>'] == gptscan.show_in_folder

    # Verify <KP_Enter> on root
    assert '<KP_Enter>' in root_bindings
    root_bindings['<KP_Enter>'](None)
    mock_on_root_return.assert_called_once()

    # Reset global GUI state to preserve test isolation
    monkeypatch.setattr(gptscan, 'filter_var', None)
    monkeypatch.setattr(gptscan, 'all_var', None)

