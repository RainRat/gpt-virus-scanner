import pytest
from unittest.mock import MagicMock, patch
import os
import json
import gptscan

@pytest.fixture
def mock_tree(monkeypatch):
    mock_tree = MagicMock()
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    def mock_item_func(item_id, option=None):
        vals = mock_tree._item_values.get(item_id, ())
        if option == "values":
            return vals
        return {"values": vals}

    mock_tree.item.side_effect = mock_item_func
    mock_tree._item_values = {}
    mock_tree.selection.return_value = []
    return mock_tree

def test_resolve_file_path_from_string_exists(tmp_path):
    f = tmp_path / "exists.py"
    f.touch()

    result = gptscan._resolve_file_path(str(f))
    assert result == str(f)

def test_resolve_file_path_from_string_not_found(monkeypatch):
    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, "messagebox", mock_msgbox)

    result = gptscan._resolve_file_path("non_existent.py")

    assert result is None
    mock_msgbox.showwarning.assert_called_with("File Not Found", "The file 'non_existent.py' could not be located.")

def test_resolve_file_path_from_selection_exists(mock_tree, tmp_path):
    f = tmp_path / "selection.py"
    f.touch()

    mock_tree.selection.return_value = ["item1"]
    raw_values = [str(f), "90%", "", "", "", "print('hi')", "1"]
    mock_tree._item_values["item1"] = list(raw_values) + [json.dumps(raw_values)]

    result = gptscan._resolve_file_path(None)
    assert result == str(f)

def test_resolve_file_path_no_selection(mock_tree):
    mock_tree.selection.return_value = []
    result = gptscan._resolve_file_path(None)
    assert result is None

def test_resolve_file_path_virtual_prefix():
    # Paths starting with '[' should bypass existence check
    path = "[Clipboard]"
    result = gptscan._resolve_file_path(path)
    assert result == path

def test_resolve_file_path_url_bypass(monkeypatch):
    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, "messagebox", mock_msgbox)

    url = "https://example.com/script.py"
    result = gptscan._resolve_file_path(url)

    assert result == url
    mock_msgbox.showwarning.assert_not_called()

def test_resolve_file_path_no_verify(tmp_path):
    path = str(tmp_path / "not_there.py")
    result = gptscan._resolve_file_path(path, verify=False)
    assert result == path
