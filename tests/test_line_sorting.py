import pytest
from unittest.mock import MagicMock
import gptscan
from tkinter import ttk

class _FakeTree:
    def __init__(self, rows, column_map=None):
        self.data = rows
        self.order = list(rows.keys())
        self.heading_calls = {}
        self.column_map = column_map or {}

    def set(self, iid, col):
        # In gptscan, tree.set(k, col) returns the value of that column
        # Our mock needs to know which index the column corresponds to
        return self.data[iid]["values"][self.column_map.get(col, 0)]

    def get_children(self, *_):
        return tuple(self.order)

    def move(self, iid, _, index):
        self.order.remove(iid)
        self.order.insert(index, iid)

    def heading(self, col, command=None):
        self.heading_calls[col] = command

def test_sort_column_line_numerical():
    # Setup data with line numbers that would sort incorrectly as strings
    # "1", "10", "2" -> sorted as strings: "1", "10", "2"
    # sorted as numbers: 1, 2, 10
    rows = {
        "item1": {"values": ("file1.py", "10", "90%")},
        "item2": {"values": ("file2.py", "2", "80%")},
        "item3": {"values": ("file3.py", "1", "100%")},
        "item4": {"values": ("file4.py", "-", "0%")}
    }
    column_map = {"path": 0, "line": 1, "own_conf": 2}
    tree = _FakeTree(rows, column_map)

    # Sort by line, ascending
    gptscan.sort_column(tree, "line", False)

    # Expected order: item4 (-1), item3 (1), item2 (2), item1 (10)
    assert tree.order == ["item4", "item3", "item2", "item1"]

    # Sort by line, descending
    gptscan.sort_column(tree, "line", True)
    assert tree.order == ["item1", "item2", "item3", "item4"]

def test_sort_column_own_conf():
    rows = {
        "item1": {"values": ("file1.py", "1", "90%")},
        "item2": {"values": ("file2.py", "2", "100%")},
        "item3": {"values": ("file3.py", "3", "80%")}
    }
    column_map = {"path": 0, "line": 1, "own_conf": 2}
    tree = _FakeTree(rows, column_map)

    # Sort by own_conf, ascending
    gptscan.sort_column(tree, "own_conf", False)
    # Expected order: item3 (80), item1 (90), item2 (100)
    assert tree.order == ["item3", "item1", "item2"]
