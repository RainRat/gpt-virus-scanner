import json
from gptscan import standardize_result_dict, parse_report_content
import pytest

def test_standardize_result_dict_none():
    result = standardize_result_dict(None)
    assert isinstance(result, dict)
    for val in result.values():
        assert val == ""

def test_standardize_result_dict_non_dict():
    result = standardize_result_dict(["not", "a", "dict"])
    assert isinstance(result, dict)
    for val in result.values():
        assert val == ""

def test_parse_report_content_with_null_elements():
    content = '[{"path": "test.py", "own_conf": "50%"}, null, {"path": "test2.py", "own_conf": "60%"}]'
    results = parse_report_content(content, filename_hint="test.json")
    assert len(results) == 2
    assert results[0]["path"] == "test.py"
    assert results[1]["path"] == "test2.py"

def test_parse_report_content_with_non_dict_elements():
    content = '[{"path": "test.py", "own_conf": "50%"}, "string", 123, {"path": "test2.py", "own_conf": "60%"}]'
    results = parse_report_content(content, filename_hint="test.json")
    assert len(results) == 2
    assert results[0]["path"] == "test.py"
    assert results[1]["path"] == "test2.py"
