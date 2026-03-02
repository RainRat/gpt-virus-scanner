import pytest
from unittest.mock import MagicMock, patch
import gptscan
import tkinter as tk

@pytest.fixture
def mock_ui_env(monkeypatch):
    mock_root = MagicMock()
    mock_tree = MagicMock()
    mock_textbox = MagicMock()
    mock_filter_entry = MagicMock()
    mock_view_button = MagicMock()
    mock_rescan_button = MagicMock()

    monkeypatch.setattr(gptscan, 'root', mock_root)
    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    monkeypatch.setattr(gptscan, 'textbox', mock_textbox)
    monkeypatch.setattr(gptscan, 'filter_entry', mock_filter_entry)
    monkeypatch.setattr(gptscan, 'view_button', mock_view_button)
    monkeypatch.setattr(gptscan, 'rescan_button', mock_rescan_button)

    # Mock __str__ for focused widget comparison
    mock_tree.__str__.return_value = ".tree"
    mock_textbox.__str__.return_value = ".textbox"
    mock_filter_entry.__str__.return_value = ".filter"

    return {
        'root': mock_root,
        'tree': mock_tree,
        'textbox': mock_textbox,
        'filter_entry': mock_filter_entry,
        'view_button': mock_view_button,
        'rescan_button': mock_rescan_button
    }

def test_on_root_return_triggers_scan(mock_ui_env, monkeypatch):
    mock_focused = MagicMock()
    mock_focused.__str__.return_value = ".some_other_widget"
    mock_ui_env['root'].focus_get.return_value = mock_focused
    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, 'button_click', mock_button_click)

    gptscan.on_root_return()

    mock_button_click.assert_called_once()

def test_on_root_return_ignored_when_tree_focused(mock_ui_env, monkeypatch):
    mock_ui_env['root'].focus_get.return_value = mock_ui_env['tree']
    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, 'button_click', mock_button_click)

    gptscan.on_root_return()

    mock_button_click.assert_not_called()

def test_on_root_return_ignored_when_textbox_focused(mock_ui_env, monkeypatch):
    mock_ui_env['root'].focus_get.return_value = mock_ui_env['textbox']
    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, 'button_click', mock_button_click)

    gptscan.on_root_return()

    mock_button_click.assert_not_called()

def test_on_root_return_ignored_when_filter_focused(mock_ui_env, monkeypatch):
    mock_ui_env['root'].focus_get.return_value = mock_ui_env['filter_entry']
    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, 'button_click', mock_button_click)

    gptscan.on_root_return()

    mock_button_click.assert_not_called()

def test_on_root_return_handles_none_root(monkeypatch):
    monkeypatch.setattr(gptscan, 'root', None)
    mock_button_click = MagicMock()
    monkeypatch.setattr(gptscan, 'button_click', mock_button_click)

    # Should just return without error
    gptscan.on_root_return()
    mock_button_click.assert_not_called()

def test_update_button_states_enabled(mock_ui_env):
    mock_ui_env['tree'].selection.return_value = ("item1",)

    gptscan.update_button_states()

    mock_ui_env['view_button'].config.assert_called_with(state="normal")
    mock_ui_env['rescan_button'].config.assert_called_with(state="normal")

def test_update_button_states_disabled(mock_ui_env):
    mock_ui_env['tree'].selection.return_value = ()

    gptscan.update_button_states()

    mock_ui_env['view_button'].config.assert_called_with(state="disabled")
    mock_ui_env['rescan_button'].config.assert_called_with(state="disabled")

def test_update_button_states_handles_none(monkeypatch):
    monkeypatch.setattr(gptscan, 'tree', None)
    # Should not raise error
    gptscan.update_button_states()
