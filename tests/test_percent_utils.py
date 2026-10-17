import pytest
from gptscan import parse_percent, format_percent

@pytest.mark.parametrize("val, default, expected", [
    ("85%", -1.0, 85.0),
    ("100%", -1.0, 100.0),
    ("0%", -1.0, 0.0),
    (" 50.5% ", -1.0, 50.5),
    ("85", -1.0, -1.0),
    ("invalid", -1.0, -1.0),
    ("", -1.0, -1.0),
    (None, -1.0, -1.0),
    (85, -1.0, -1.0),
    ("abc%", 0.0, 0.0),
])
def test_parse_percent(val, default, expected):
    """Test percentage string parsing across various inputs and fallback defaults."""
    assert parse_percent(val, default) == expected


@pytest.mark.parametrize("val, expected", [
    (85, "85%"),
    (85.0, "85%"),
    ("85", "85%"),
    (0, "0%"),
    (100, "100%"),
    (50.5, "50%"),
    ("invalid", "Error"),
    (None, "Error"),
])
def test_format_percent(val, expected):
    """Test formatting numeric threat levels into percentage strings."""
    assert format_percent(val) == expected
