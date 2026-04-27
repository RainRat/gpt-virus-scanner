import pytest
from gptscan import standardize_result_dict

def test_standardize_result_dict_fallback():
    # Item with an empty 'path' key but a valid 'File Path' key
    item = {
        "path": "",
        "File Path": "correct/path.py",
        "Confidence": "85%"
    }

    standardized = standardize_result_dict(item)

    # Before the fix, it would have picked the empty string from 'path'
    assert standardized["path"] == "correct/path.py"
    # own_conf should pick up 'Confidence'
    assert standardized["own_conf"] == "85%"

def test_standardize_result_dict_none_handling():
    # Item with None value for a key
    item = {
        "path": None,
        "uri": "path/via/uri.py"
    }

    standardized = standardize_result_dict(item)
    assert standardized["path"] == "path/via/uri.py"
