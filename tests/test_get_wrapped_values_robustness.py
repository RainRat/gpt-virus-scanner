import pytest
from gptscan import get_wrapped_values

def test_get_wrapped_values_with_generator():
    """Verify that get_wrapped_values correctly handles a generator and doesn't exhaust it prematurely."""

    def my_generator():
        yield "path/to/file.py"
        yield "80%"
        yield "Admin notes"
        yield "User notes"
        yield "90%"
        yield "print('hello')"
        yield "10"
        yield '{"some": "json"}'

    # We provide col_widths and measure so the function doesn't need a real Treeview or Tkinter font
    col_widths = [100, 50, 150, 150, 50, 200, 30]
    measure = lambda x: len(x) * 7 # Dummy measure function

    # Act
    # Tree can be None if col_widths and measure are provided
    result = get_wrapped_values(None, my_generator(), measure=measure, col_widths=col_widths)

    # Assert
    assert len(result) == 8
    assert result[0] == "path/to/file.py"
    assert result[4] == "90%"
    assert result[5] == "print('hello')"
    assert result[6] == "10"
    assert result[7] == '{"some": "json"}'

def test_get_wrapped_values_short_iterable():
    """Verify it handles iterables shorter than 6 elements."""
    def short_gen():
        yield "only"
        yield "two"

    col_widths = [100, 100]
    measure = lambda x: len(x) * 7

    result = get_wrapped_values(None, short_gen(), measure=measure, col_widths=col_widths)
    assert len(result) == 2
    assert result == ["only", "two"]
