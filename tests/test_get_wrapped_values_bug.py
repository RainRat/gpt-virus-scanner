import pytest
from unittest.mock import MagicMock
from gptscan import get_wrapped_values

def test_get_wrapped_values_with_generator():
    """Verify that get_wrapped_values correctly handles a generator without TypeError or exhaustion."""
    mock_tree = MagicMock()
    # Mock tree['columns']
    mock_tree.__getitem__.return_value = ["c1", "c2", "c3", "c4", "c5", "c6", "c7"]
    # Mock tree.column(cid)['width']
    mock_tree.column.return_value = {'width': 100}

    def mock_measure(text):
        return len(str(text)) * 10

    # Generator with 8 items
    def gen():
        for i in range(1, 9):
            yield f"item{i}"

    # This is expected to FAIL before the fix
    result = get_wrapped_values(mock_tree, gen(), measure=mock_measure)

    # If it didn't raise TypeError, it might still fail due to exhaustion
    assert len(result) == 8
    assert result[0].startswith("item1") # adjust_newlines might add wrap markers
    assert result[7] == "item8"
