import tkinter.font
import pytest
from unittest.mock import MagicMock
from gptscan import Config, load_file, parse_percent, adjust_newlines, sort_column


# --- Config Tests ---

def test_config_set_extensions():
    extensions = [".py", ".js", ".BAT"]
    Config.set_extensions(extensions, missing=False)

    assert Config.extensions == extensions
    assert Config.extensions_set == {".py", ".js", ".bat"}
    assert Config.extensions_missing is False

def test_config_set_extensions_empty():
    Config.set_extensions([], missing=True)
    assert Config.extensions == []
    assert Config.extensions_set == set()
    assert Config.extensions_missing is True

# --- load_file Tests ---

def test_load_file_single_line(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("line 1\nline 2")

    result = load_file(str(f), mode='single_line')
    assert result == "line 1"

def test_load_file_multi_line(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("line 1\nline 2")

    result = load_file(str(f), mode='multi_line')
    assert result == ["line 1", "line 2"]

def test_load_file_missing_single_line():
    result = load_file("non_existent.txt", mode='single_line')
    assert result == ""

def test_load_file_missing_multi_line():
    result = load_file("non_existent.txt", mode='multi_line')
    assert result == []

# --- parse_percent Tests ---

@pytest.mark.parametrize("input_val, expected", [
    ("50%", 50.0),
    ("100%", 100.0),
    ("0%", 0.0),
    ("  75%  ", 75.0),
    ("invalid", -1.0),
    ("", -1.0),
    (None, -1.0),
    ("50", -1.0) # Missing %
])
def test_parse_percent(input_val, expected):
    assert parse_percent(input_val, default=-1.0) == expected

def test_parse_percent_custom_default():
    assert parse_percent("invalid", default=0.0) == 0.0

# --- adjust_newlines Tests ---

def test_adjust_newlines_no_wrap(monkeypatch):
    # Mock font measure
    monkeypatch.setattr(tkinter.font.Font, 'measure', lambda self, text: len(text) * 10)

    text = "hello world"
    # width 200 > 110
    result = adjust_newlines(text, width=200, measure=lambda t: len(t) * 10)
    assert result == "hello world"

def test_adjust_newlines_wrap(monkeypatch):
    monkeypatch.setattr(tkinter.font.Font, 'measure', lambda self, text: len(text) * 10)

    text = "hello world"
    # width 60. 'hello ' is 60 (including space). 'world' needs wrap.
    # Logic is: check 'hello world'. len 11 * 10 = 110. > width-pad (60-10=50).
    # so 'hello' stays, 'world' wraps.

    result = adjust_newlines(text, width=60, measure=lambda t: len(t) * 10)
    assert result == "hello\nworld"

def test_adjust_newlines_non_string():
    assert adjust_newlines(123, 100) == 123

# --- sort_column Tests ---

def test_sort_column_numeric(monkeypatch):
    tv = MagicMock()
    # Mock get_children and set
    tv.get_children.return_value = ["i1", "i2", "i3"]

    data = {
        "i1": "10%",
        "i2": "50%",
        "i3": "5%"
    }
    tv.set.side_effect = lambda k, c: data[k]

    # Sort 'own_conf' which uses parse_percent
    sort_column(tv, "own_conf", reverse=False)

    # Check move calls.
    # Order should be i3 (5%), i1 (10%), i2 (50%)
    tv.move.assert_any_call("i3", "", 0)
    tv.move.assert_any_call("i1", "", 1)
    tv.move.assert_any_call("i2", "", 2)

def test_sort_column_text(monkeypatch):
    tv = MagicMock()
    tv.get_children.return_value = ["i1", "i2"]

    data = {
        "i1": "bravo",
        "i2": "alpha"
    }
    tv.set.side_effect = lambda k, c: data[k]

    sort_column(tv, "path", reverse=False)

    # Order: alpha, bravo -> i2, i1
    tv.move.assert_any_call("i2", "", 0)
    tv.move.assert_any_call("i1", "", 1)
