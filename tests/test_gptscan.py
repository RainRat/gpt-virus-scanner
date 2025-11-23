import json
from types import SimpleNamespace
from pathlib import Path

import pytest

import gptscan


def test_load_file_single_line(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("first line\nsecond line")

    result = gptscan.load_file(str(file_path))

    assert result == "first line"


def test_load_file_multi_line(tmp_path):
    file_path = tmp_path / "lines.txt"
    file_path.write_text("line1\nline2\n")

    result = gptscan.load_file(str(file_path), mode="multi_line")

    assert result == ["line1", "line2"]


def test_load_file_missing_returns_empty_string():
    assert gptscan.load_file("non_existent_file.txt") == ""


class _MockChoice:
    def __init__(self, content: str):
        self.message = SimpleNamespace(content=content)


class _MockResponse:
    def __init__(self, content: str):
        self.choices = [_MockChoice(content)]


def test_extract_data_from_gpt_response_parses_json():
    response = _MockResponse(
        json.dumps(
            {
                "administrator": "Admin",
                "end-user": "User",
                "threat-level": 75,
            }
        )
    )

    parsed = gptscan.extract_data_from_gpt_response(response)

    assert parsed["administrator"] == "Admin"
    assert parsed["end-user"] == "User"
    assert parsed["threat-level"] == 75


def test_extract_data_from_gpt_response_invalid_missing_keys():
    response = _MockResponse(json.dumps({"administrator": "Only admin"}))

    error = gptscan.extract_data_from_gpt_response(response)

    assert "Missing keys" in error


def test_extract_data_from_gpt_response_invalid_threat_level():
    response = _MockResponse(
        json.dumps(
            {
                "administrator": "Admin",
                "end-user": "User",
                "threat-level": "high",  # not an integer
            }
        )
    )

    error = gptscan.extract_data_from_gpt_response(response)

    assert "not a valid integer" in error


def test_list_files_returns_files_only(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "root.txt").write_text("root")
    (nested / "child.txt").write_text("child")
    (nested / "subdir").mkdir()

    files = gptscan.list_files(tmp_path)

    assert set(Path(f) for f in files) == {
        tmp_path / "root.txt",
        nested / "child.txt",
    }


def test_sort_column_orders_treeview():
    class _FakeTree:
        def __init__(self):
            self.data = {
                "item1": {"values": ("b", "10%")},
                "item2": {"values": ("a", "30%")},
            }
            self.order = ["item1", "item2"]
            self.heading_calls = {}

        def set(self, iid, col):
            columns = {"name": 0, "own_conf": 1}
            return self.data[iid]["values"][columns[col]]

        def get_children(self, *_):
            return tuple(self.order)

        def move(self, iid, _, index):
            self.order.remove(iid)
            self.order.insert(index, iid)

        def heading(self, col, command=None):
            self.heading_calls[col] = command

    tree = _FakeTree()

    gptscan.sort_column(tree, "name", False)
    assert tree.get_children("") == ("item2", "item1")

    gptscan.sort_column(tree, "own_conf", False)
    assert tree.get_children("") == ("item1", "item2")
