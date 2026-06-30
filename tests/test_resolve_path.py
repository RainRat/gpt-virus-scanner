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

def test_resolve_file_paths_from_string_exists(tmp_path):
    f = tmp_path / "exists.py"
    f.touch()

    result = gptscan._resolve_file_paths(str(f))
    assert result == [str(f)]

def test_resolve_file_paths_from_string_not_found(monkeypatch):
    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, "messagebox", mock_msgbox)

    result = gptscan._resolve_file_paths("non_existent.py")

    assert result == []
    mock_msgbox.showwarning.assert_called_with("Files Not Found", "The selected file(s) could not be located on disk.")

def test_resolve_file_paths_from_selection_exists(mock_tree, tmp_path):
    f = tmp_path / "selection.py"
    f.touch()

    mock_tree.selection.return_value = ["item1"]
    raw_values = [str(f), "90%", "", "", "", "print('hi')", "1"]
    mock_tree._item_values["item1"] = list(raw_values) + [json.dumps(raw_values)]

    result = gptscan._resolve_file_paths(None)
    assert result == [str(f)]

def test_resolve_file_paths_no_selection(mock_tree):
    mock_tree.selection.return_value = []
    result = gptscan._resolve_file_paths(None)
    assert result == []

def test_resolve_file_paths_virtual_prefix():
    # Paths starting with '[' should bypass existence check
    path = "[Clipboard]"
    result = gptscan._resolve_file_paths(path)
    assert result == [path]

def test_resolve_file_paths_url_bypass(monkeypatch):
    mock_msgbox = MagicMock()
    monkeypatch.setattr(gptscan, "messagebox", mock_msgbox)

    url = "https://example.com/script.py"
    result = gptscan._resolve_file_paths(url)

    assert result == [url]
    mock_msgbox.showwarning.assert_not_called()

def test_resolve_file_paths_no_verify(tmp_path):
    path = str(tmp_path / "not_there.py")
    result = gptscan._resolve_file_paths(path, verify=False)
    assert result == [path]

def test_resolve_file_paths_multi_selection(mock_tree, tmp_path):
    f1 = tmp_path / "f1.py"
    f1.touch()
    f2 = tmp_path / "f2.py"
    f2.touch()

    mock_tree.selection.return_value = ["item1", "item2"]
    val1 = [str(f1), "50%", "", "", "", "print(1)", "1"]
    val2 = [str(f2), "50%", "", "", "", "print(2)", "1"]
    mock_tree._item_values["item1"] = val1 + [json.dumps(val1)]
    mock_tree._item_values["item2"] = val2 + [json.dumps(val2)]

    result = gptscan._resolve_file_paths(None)
    assert result == [str(f1), str(f2)]
