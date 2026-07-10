import pytest
from unittest.mock import MagicMock, patch
from typing import Iterable, Any
from gptscan import get_wrapped_values
import json

def test_get_wrapped_values_with_generator(monkeypatch):
    """Verify that get_wrapped_values handles generators correctly without raising TypeError."""
    mock_tree = MagicMock()
    # Mock tree['columns'] to return a list of 7 column names
    mock_tree.__getitem__.side_effect = lambda key: ("c1", "c2", "c3", "c4", "c5", "c6", "orig_json") if key == "columns" else MagicMock()
    mock_tree.column.return_value = {"width": 100}

    # Mock font measurement
    monkeypatch.setattr("tkinter.font.Font", MagicMock())

    # Create a generator for input values
    raw_data = ["path/to/file", "50%", "admin", "user", "40%", "snippet", '{"key": "value"}']
    gen_values = (v for v in raw_data)

    # This should not raise TypeError: object of type 'generator' has no len()
    try:
        result = get_wrapped_values(mock_tree, gen_values)
    except TypeError as e:
        pytest.fail(f"get_wrapped_values raised TypeError with generator: {e}")

    assert len(result) == 7
    # Verify content (assuming adjust_newlines might have modified them, but they should be present)
    assert "path/to/file" in result[0]
    assert result[6] == '{"key": "value"}'

def test_get_wrapped_values_preserves_extra_values(monkeypatch):
    """Verify that values beyond index 6 are preserved correctly when using a generator."""
    mock_tree = MagicMock()
    mock_tree.__getitem__.side_effect = lambda key: ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8") if key == "columns" else MagicMock()
    mock_tree.column.return_value = {"width": 100}
    monkeypatch.setattr("tkinter.font.Font", MagicMock())

    raw_data = ["v1", "v2", "v3", "v4", "v5", "v6", "extra1", "extra2"]
    gen_values = (v for v in raw_data)

    result = get_wrapped_values(mock_tree, gen_values)

    assert len(result) == 8
    assert result[6] == "extra1"
    assert result[7] == "extra2"
