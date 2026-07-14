import pytest
from unittest.mock import MagicMock
from gptscan import get_wrapped_values

def test_get_wrapped_values_with_list():
    tree = MagicMock()
    tree.column.return_value = {'width': 100}
    tree.__getitem__.return_value = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7']

    values = ["val1", "val2", "val3", "val4", "val5", "val6", "val7", "val8"]

    # Mock measure to just return the value as is (no wrapping)
    measure = lambda x: len(x)

    result = get_wrapped_values(tree, values, measure=measure)

    assert result == values
    assert len(result) == 8

def test_get_wrapped_values_with_generator():
    tree = MagicMock()
    tree.column.return_value = {'width': 100}
    tree.__getitem__.return_value = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7']

    def gen():
        yield from ["val1", "val2", "val3", "val4", "val5", "val6", "val7", "val8"]

    measure = lambda x: len(x)

    # This would fail before the fix with "TypeError: object of type 'generator' has no len()"
    result = get_wrapped_values(tree, gen(), measure=measure)

    assert result == ["val1", "val2", "val3", "val4", "val5", "val6", "val7", "val8"]
    assert len(result) == 8

def test_get_wrapped_values_short_iterable():
    tree = MagicMock()
    tree.column.return_value = {'width': 100}
    tree.__getitem__.return_value = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7']

    values = ["val1", "val2"]
    measure = lambda x: len(x)

    result = get_wrapped_values(tree, values, measure=measure)

    assert result == ["val1", "val2"]
    assert len(result) == 2
