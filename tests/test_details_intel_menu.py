import pytest
import json
import tkinter as tk
from unittest.mock import MagicMock, patch
import gptscan
from tests.test_view_details import mock_view_details_env, setup_details

def test_details_intel_menu_init(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    # Setup a result
    raw = ["test.py", "90%", "Admin", "User", "80%", "print('hello')", 1]
    mock_tree._item_values["item1"] = ["test.py", "90%", "Admin", "User", "80%", "print('hello')", 1, json.dumps(raw)]
    mock_tree.get_children.return_value = ["item1"]
    mock_tree.selection.return_value = ["item1"]

    gptscan.view_details(item_id="item1")

    # Check if Intel Menubutton exists
    assert "menubtn_Intel" in captured
    intel_btn_mock, intel_btn_menu = captured["menubtn_Intel"]

    # Check menu items
    menu_items = captured.get("menu_items_" + str(id(intel_btn_menu)), [])
    labels = [item[0] for item in menu_items]
    assert "Check on VirusTotal" in labels
    assert "View Online" in labels

def test_details_intel_menu_state_virtual(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    # Setup a virtual result (inside archive)
    raw = ["[ZIP] test.py", "90%", "Admin", "User", "80%", "print('hello')", 1]
    mock_tree._item_values["item1"] = ["[ZIP] test.py", "90%", "Admin", "User", "80%", "print('hello')", 1, json.dumps(raw)]
    mock_tree.get_children.return_value = ["item1"]
    mock_tree.selection.return_value = ["item1"]

    gptscan.view_details(item_id="item1")

    intel_btn_mock, intel_btn_menu = captured["menubtn_Intel"]

    # Check if 'View Online' is disabled for non-URL virtual paths
    # We need to find the entryconfig calls for this menu
    entryconfigs = captured.get("menu_entryconfigs_" + str(id(intel_btn_menu)), {})
    assert entryconfigs.get("View Online", {}).get("state") == "disabled"
    assert entryconfigs.get("Check on VirusTotal", {}).get("state") == "normal"

def test_details_intel_menu_state_url(mock_view_details_env):
    captured, mock_msgbox, mock_tree, mock_toplevel = mock_view_details_env
    # Setup a URL result
    raw = ["[URL] https://github.com/user/repo/file.py", "90%", "Admin", "User", "80%", "print('hello')", 1]
    mock_tree._item_values["item1"] = ["[URL] https://github.com/user/repo/file.py", "90%", "Admin", "User", "80%", "print('hello')", 1, json.dumps(raw)]
    mock_tree.get_children.return_value = ["item1"]
    mock_tree.selection.return_value = ["item1"]

    gptscan.view_details(item_id="item1")

    intel_btn_mock, intel_btn_menu = captured["menubtn_Intel"]

    # Check if 'View Online' is enabled for URL paths
    entryconfigs = captured.get("menu_entryconfigs_" + str(id(intel_btn_menu)), {})
    assert entryconfigs.get("View Online", {}).get("state") == "normal"
    assert entryconfigs.get("Check on VirusTotal", {}).get("state") == "normal"
