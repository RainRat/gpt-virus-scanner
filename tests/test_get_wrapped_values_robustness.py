import tkinter as tk
from tkinter import ttk
from unittest.mock import MagicMock
import pytest
import gptscan

def test_get_wrapped_values_with_generator(monkeypatch):
    # Mock font measure to avoid needing a real display
    mock_measure = lambda text: len(text) * 10
    monkeypatch.setattr(gptscan, "default_font_measure", mock_measure)

    # Mock tree
    mock_tree = MagicMock()
    mock_tree.column.return_value = {'width': 100}
    mock_tree.__getitem__.side_effect = lambda key: ("col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8") if key == "columns" else MagicMock()

    # Generator for values
    def my_gen():
        for i in range(8):
            yield f"value {i}"

    # In the unfixed state, this raises TypeError: object of type 'generator' has no len()
    res = gptscan.get_wrapped_values(mock_tree, my_gen(), measure=mock_measure)
    assert len(res) == 8
    assert res[0] == "value 0"
