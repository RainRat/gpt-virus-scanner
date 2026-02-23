import pytest
from unittest.mock import MagicMock, patch
import gptscan
import os
import json
import hashlib

@pytest.fixture
def mock_tree(monkeypatch):
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree, raising=False)
    monkeypatch.setattr(gptscan, 'root', MagicMock(), raising=False)

    def mock_item_func(item_id, option=None):
        vals = mock_tree._item_values.get(item_id, ())
        if option == "values":
            return vals
        return {"values": vals}

    mock_tree.item.side_effect = mock_item_func
    mock_tree._item_values = {}
    return mock_tree

def test_get_file_sha256(tmp_path):
    d = tmp_path / "test"
    d.mkdir()
    f = d / "hello.txt"
    content = b"hello world"
    f.write_bytes(content)

    expected_hash = hashlib.sha256(content).hexdigest()
    assert gptscan.get_file_sha256(str(f)) == expected_hash

def test_get_file_sha256_not_found():
    assert gptscan.get_file_sha256("non_existent_file") == ""

def test_copy_sha256(mock_tree, tmp_path):
    f = tmp_path / "test.py"
    content = b"print('hello')"
    f.write_bytes(content)
    expected_hash = hashlib.sha256(content).hexdigest()

    mock_tree.selection.return_value = ["item1"]
    raw_values = [str(f), "90%", "Admin", "User", "80%", "print('hello')"]
    mock_tree._item_values["item1"] = (str(f), "90%", "Admin", "User", "80%", "print('hello')", json.dumps(raw_values))

    with patch('gptscan.update_status') as mock_status:
        gptscan.copy_sha256()

    mock_tree.clipboard_clear.assert_called_once()
    mock_tree.clipboard_append.assert_called_with(expected_hash)
    mock_status.assert_called()

def test_check_virustotal(mock_tree, tmp_path):
    f = tmp_path / "test.py"
    content = b"print('hello')"
    f.write_bytes(content)
    expected_hash = hashlib.sha256(content).hexdigest()

    mock_tree.selection.return_value = ["item1"]
    raw_values = [str(f), "90%", "Admin", "User", "80%", "print('hello')"]
    mock_tree._item_values["item1"] = (str(f), "90%", "Admin", "User", "80%", "print('hello')", json.dumps(raw_values))

    with patch('webbrowser.open') as mock_open, patch('gptscan.update_status') as mock_status:
        gptscan.check_virustotal()

    expected_url = f"https://www.virustotal.com/gui/file/{expected_hash}"
    mock_open.assert_called_once_with(expected_url)
    mock_status.assert_called()

def test_check_virustotal_with_path(tmp_path):
    f = tmp_path / "test.py"
    content = b"print('hello')"
    f.write_bytes(content)
    expected_hash = hashlib.sha256(content).hexdigest()

    with patch('webbrowser.open') as mock_open, patch('gptscan.update_status') as mock_status:
        gptscan.check_virustotal(str(f))

    expected_url = f"https://www.virustotal.com/gui/file/{expected_hash}"
    mock_open.assert_called_once_with(expected_url)

def test_check_virustotal_not_found(monkeypatch):
    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, 'messagebox', mock_msgbox)

    gptscan.check_virustotal("ghost.py")
    mock_msgbox.showwarning.assert_called_with("File Not Found", "The file 'ghost.py' could not be located.")
