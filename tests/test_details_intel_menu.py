import pytest
from unittest.mock import MagicMock, patch, mock_open
import gptscan
import json
import tkinter as tk

@pytest.fixture
def mock_details_env(monkeypatch):
    """Setup a mock environment specifically for testing the new Intel menu in view_details."""
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    mock_tree._item_values = {}
    def mock_item_func(item_id, option=None):
        vals = mock_tree._item_values.get(item_id, [])
        if option == "values": return vals
        return {"values": vals}
    mock_tree.item.side_effect = mock_item_func
    mock_tree.exists.side_effect = lambda iid: iid in mock_tree._item_values

    mock_root = MagicMock()
    monkeypatch.setattr(gptscan, 'root', mock_root)
    monkeypatch.setattr(gptscan.tk, 'Toplevel', MagicMock())

    captured = {'menus': [], 'buttons': {}, 'menubuttons': {}}

    def mock_button_init(master, **kwargs):
        btn = MagicMock()
        text = kwargs.get('text', '')
        if text: captured['buttons'][text] = (btn, kwargs.get('command'))
        return btn
    monkeypatch.setattr(gptscan.ttk, 'Button', mock_button_init)

    def mock_menubutton_init(master, **kwargs):
        mb = MagicMock()
        text = kwargs.get('text', '')
        if text: captured['menubuttons'][text] = mb
        return mb
    monkeypatch.setattr(gptscan.ttk, 'Menubutton', mock_menubutton_init)

    class MockMenu:
        def __init__(self, master=None, **kwargs):
            self.master = master
            self.items = {}
            self.configs = {}
            captured['menus'].append(self)
        def add_command(self, **kwargs):
            label = kwargs.get('label')
            if label: self.items[label] = kwargs.get('command')
        def entryconfig(self, label, **kwargs):
            if label in self.items:
                self.configs[label] = kwargs
        def add_separator(self): pass
    monkeypatch.setattr(gptscan.tk, 'Menu', MockMenu)

    # Mock other necessary widgets
    monkeypatch.setattr(gptscan.ttk, 'Entry', MagicMock())
    monkeypatch.setattr(gptscan.tk, 'Label', MagicMock())
    monkeypatch.setattr(gptscan.ttk, 'Label', MagicMock())
    monkeypatch.setattr(gptscan.scrolledtext, 'ScrolledText', MagicMock())

    return captured, mock_tree

def test_intel_menu_creation_and_state(mock_details_env, monkeypatch):
    captured, mock_tree = mock_details_env
    path = "test.py"
    raw_vals = [path, "90%", "Admin", "User", "80%", "snippet", 1]
    mock_tree._item_values["item1"] = raw_vals + [json.dumps(raw_vals)]
    mock_tree.get_children.return_value = ["item1"]

    gptscan.view_details(item_id="item1")

    # Verify Intel Menubutton exists
    assert "Intel" in captured['menubuttons']

    # Verify the inner menu was created and has items
    intel_menu = None
    for menu in captured['menus']:
        if "Check on VirusTotal" in menu.items:
            intel_menu = menu
            break

    assert intel_menu is not None
    assert "Check on VirusTotal" in intel_menu.items
    assert "View Online" in intel_menu.items

    # Verify Open and Show in Folder are now buttons (not just in header)
    assert "Open" in captured['buttons']
    assert "Show in Folder" in captured['buttons']

    # Verify states in refresh_content for a local file
    # We need to find the intel menu again as it might have been updated
    assert intel_menu.configs["View Online"]["state"] == "normal"

    # Test with a virtual path (e.g. from clipboard)
    path_v = "[Clipboard]"
    raw_vals_v = [path_v, "90%", "", "", "", "snippet", 1]
    mock_tree._item_values["item2"] = raw_vals_v + [json.dumps(raw_vals_v)]
    mock_tree.get_children.return_value = ["item1", "item2"]

    # Trigger refresh by navigating to item2
    # In our mock, we can just call view_details again with item2
    gptscan.view_details(item_id="item2")

    # Check that View Online is disabled for non-URL virtual paths
    # find the new menu created for the second call
    # Note: refresh_content uses the menu defined in view_details scope
    intel_menu_v = None
    for menu in captured['menus']:
        if "Check on VirusTotal" in menu.items and "View Online" in menu.configs:
            intel_menu_v = menu
            # We want the most recent config

    assert intel_menu_v is not None
    assert intel_menu_v.configs["View Online"]["state"] == "disabled"

    # Verify Open and Reveal are disabled for virtual paths
    open_btn, _ = captured['buttons']["Open"]
    reveal_btn, _ = captured['buttons']["Show in Folder"]
    # We need to check if config was called with state='disabled'
    open_btn.config.assert_called_with(state='disabled')
    reveal_btn.config.assert_called_with(state='disabled')

def test_intel_menu_shortcuts(mock_details_env):
    captured, mock_tree = mock_details_env
    mock_toplevel = gptscan.tk.Toplevel.return_value
    path = "test.py"
    raw_vals = [path, "90%", "", "", "", "snippet", 1]
    mock_tree._item_values["item1"] = raw_vals + [json.dumps(raw_vals)]
    mock_tree.get_children.return_value = ["item1"]

    captured_bindings = {}
    mock_toplevel.bind.side_effect = lambda event, func: captured_bindings.update({event: func})

    gptscan.view_details(item_id="item1")

    assert '<Control-t>' in captured_bindings
    assert '<Command-t>' in captured_bindings
    assert '<Control-l>' in captured_bindings
    assert '<Command-l>' in captured_bindings
