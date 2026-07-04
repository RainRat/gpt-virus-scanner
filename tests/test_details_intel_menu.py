import pytest
from unittest.mock import MagicMock, patch, ANY
import gptscan
import tkinter as tk
import json

@pytest.fixture
def mock_details_gui(monkeypatch):
    # Mocking necessary components within view_details scope or related
    mock_tree = MagicMock()
    mock_tree.get_children.return_value = ["item1"]
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    # We need to mock _get_item_raw_values as it's used in view_details
    monkeypatch.setattr(gptscan, '_get_item_raw_values', MagicMock(return_value=["test.py", "50%", "admin", "user", "80%", "print(1)", "1"]))

    # Mock other GUI globals
    monkeypatch.setattr(gptscan, 'root', MagicMock())
    monkeypatch.setattr(gptscan, 'current_cancel_event', None)

    return mock_tree

def test_view_details_intel_menu_creation(mock_details_gui, monkeypatch):
    """Test that the Intel Menubutton and its menu are created in the details window."""
    # We need to intercept the creation of Menubutton and Menu
    mock_menubutton = MagicMock()
    mock_menu = MagicMock()

    with patch('tkinter.ttk.Menubutton', return_value=mock_menubutton), \
         patch('tkinter.Menu', return_value=mock_menu), \
         patch('tkinter.Toplevel'), \
         patch('tkinter.ttk.Separator'), \
         patch('tkinter.ttk.Button'), \
         patch('tkinter.Label'), \
         patch('tkinter.ttk.Label'), \
         patch('tkinter.ttk.Entry'), \
         patch('tkinter.ttk.Frame'), \
         patch('tkinter.ttk.LabelFrame'), \
         patch('tkinter.ttk.Panedwindow'), \
         patch('tkinter.scrolledtext.ScrolledText'):

        gptscan.view_details(item_id="item1")

        # Verify menu items were added
        mock_menu.add_command.assert_any_call(label="Check on VirusTotal", command=ANY, accelerator="Ctrl+T")
        mock_menu.add_command.assert_any_call(label="View Online", command=ANY, accelerator="Ctrl+L")

def test_view_details_intel_menu_states_virtual(mock_details_gui, monkeypatch):
    """Test that View Online in details Intel menu is disabled for virtual paths."""
    mock_menu = MagicMock()

    # Return [Clipboard] virtual path
    gptscan._get_item_raw_values.return_value = ["[Clipboard]", "0%", "", "", "", "snippet", "-"]

    with patch('tkinter.Menu', return_value=mock_menu), \
         patch('tkinter.Toplevel'), \
         patch('tkinter.ttk.Menubutton'), \
         patch('tkinter.ttk.Separator'), \
         patch('tkinter.ttk.Button'), \
         patch('tkinter.Label'), \
         patch('tkinter.ttk.Label'), \
         patch('tkinter.ttk.Entry'), \
         patch('tkinter.ttk.Frame'), \
         patch('tkinter.ttk.LabelFrame'), \
         patch('tkinter.ttk.Panedwindow'), \
         patch('tkinter.scrolledtext.ScrolledText'):

        gptscan.view_details(item_id="item1")

        # The refresh_content is called inside view_details
        mock_menu.entryconfig.assert_any_call("View Online", state='disabled')
