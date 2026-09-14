import pytest
from unittest.mock import MagicMock, patch
import gptscan
import os

@pytest.fixture
def mock_gui_vars(monkeypatch):
    mock_tree = MagicMock()
    mock_status = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree)
    monkeypatch.setattr(gptscan, 'status_label', mock_status)
    return mock_tree, mock_status

def test_view_online_multi_selection(mock_gui_vars):
    mock_tree, _ = mock_gui_vars
    mock_tree.selection.return_value = ['item1', 'item2']

    def mock_get_item_raw_values(item_id):
        if item_id == 'item1':
            return ["file1.py", "50%", "Admin", "User", "50%", "snippet1", "10"]
        if item_id == 'item2':
            return ["file2.py", "60%", "Admin", "User", "60%", "snippet2", "20"]
        return None

    with patch('gptscan._get_item_raw_values', side_effect=mock_get_item_raw_values), \
         patch('gptscan.get_online_url', side_effect=lambda p, l: f"http://online/{p}"), \
         patch('webbrowser.open') as mock_open, \
         patch('gptscan.update_status') as mock_status_update:

        gptscan.view_online()

        assert mock_open.call_count == 2
        mock_open.assert_any_call("http://online/file1.py")
        mock_open.assert_any_call("http://online/file2.py")
        mock_status_update.assert_called_with("Opening online view for 2 files...")

def test_view_online_multi_selection_with_confirmation_yes(mock_gui_vars):
    mock_tree, _ = mock_gui_vars
    mock_tree.selection.return_value = ['item1', 'item2', 'item3', 'item4', 'item5', 'item6']

    with patch('gptscan._get_item_raw_values', return_value=["file.py", "50%", "", "", "", "snippet", "1"]), \
         patch('gptscan.messagebox.askyesno', return_value=True) as mock_ask, \
         patch('gptscan.get_online_url', return_value="http://online"), \
         patch('webbrowser.open') as mock_open:

        gptscan.view_online()

        mock_ask.assert_called_once()
        assert mock_open.call_count == 6

def test_view_online_multi_selection_with_confirmation_no(mock_gui_vars):
    mock_tree, _ = mock_gui_vars
    mock_tree.selection.return_value = ['item1', 'item2', 'item3', 'item4', 'item5', 'item6']

    with patch('gptscan.messagebox.askyesno', return_value=False) as mock_ask, \
         patch('webbrowser.open') as mock_open:

        gptscan.view_online()

        mock_ask.assert_called_once()
        assert mock_open.call_count == 0

def test_view_online_single_selection_status(mock_gui_vars):
    mock_tree, _ = mock_gui_vars
    mock_tree.selection.return_value = ['item1']

    with patch('gptscan._get_item_raw_values', return_value=["path/to/my_file.py", "50%", "", "", "", "snippet", "1"]), \
         patch('gptscan.get_online_url', return_value="http://online"), \
         patch('webbrowser.open') as mock_open, \
         patch('gptscan.update_status') as mock_status_update:

        gptscan.view_online()

        assert mock_open.call_count == 1
        mock_status_update.assert_called_with("Opening online view for my_file.py...")

def test_view_online_no_tree(monkeypatch):
    monkeypatch.setattr(gptscan, 'tree', None)
    with patch('webbrowser.open') as mock_open:
        gptscan.view_online()
        mock_open.assert_not_called()

def test_view_online_empty_selection(mock_gui_vars):
    mock_tree, _ = mock_gui_vars
    mock_tree.selection.return_value = []
    with patch('webbrowser.open') as mock_open:
        gptscan.view_online()
        mock_open.assert_not_called()

def test_view_online_string_path_with_explicit_line():
    with patch('gptscan.get_online_url') as mock_get_url, \
         patch('webbrowser.open') as mock_open, \
         patch('gptscan.update_status'):
        mock_get_url.return_value = "https://github.com/user/repo/blob/main/file.py#L42"
        gptscan.view_online("file.py", line=42)
        mock_get_url.assert_called_once_with("file.py", 42)
        mock_open.assert_called_once_with("https://github.com/user/repo/blob/main/file.py#L42")

def test_view_online_string_path_with_default_line():
    with patch('gptscan.get_online_url') as mock_get_url, \
         patch('webbrowser.open') as mock_open, \
         patch('gptscan.update_status'):
        mock_get_url.return_value = "https://github.com/user/repo/blob/main/file.py#L1"
        gptscan.view_online("file.py")
        mock_get_url.assert_called_once_with("file.py", 1)
        mock_open.assert_called_once_with("https://github.com/user/repo/blob/main/file.py#L1")

def test_view_online_string_path_unresolved_no_messagebox():
    with patch('gptscan.get_online_url', return_value=None), \
         patch('webbrowser.open') as mock_open, \
         patch('gptscan.messagebox.showinfo') as mock_showinfo:
        gptscan.view_online("file.py")
        mock_open.assert_not_called()
        mock_showinfo.assert_not_called()

def test_view_online_selection_unresolved_shows_messagebox(mock_gui_vars):
    mock_tree, _ = mock_gui_vars
    mock_tree.selection.return_value = ['item1']
    with patch('gptscan._get_item_raw_values', return_value=["local_file.py", "50%", "", "", "", "snippet", "10"]), \
         patch('gptscan.get_online_url', return_value=None), \
         patch('webbrowser.open') as mock_open, \
         patch('gptscan.messagebox.showinfo') as mock_showinfo:
        gptscan.view_online()
        mock_open.assert_not_called()
        mock_showinfo.assert_called_once()

def test_view_online_selection_dash_line_number_fallback(mock_gui_vars):
    mock_tree, _ = mock_gui_vars
    mock_tree.selection.return_value = ['item1']
    with patch('gptscan._get_item_raw_values', return_value=["file.py", "50%", "", "", "", "snippet", "-"]), \
         patch('gptscan.get_online_url') as mock_get_url, \
         patch('webbrowser.open'), \
         patch('gptscan.update_status'):
        mock_get_url.return_value = "http://online/file.py"
        gptscan.view_online()
        mock_get_url.assert_called_once_with("file.py", 1)
