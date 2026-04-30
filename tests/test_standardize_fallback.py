import pytest
from gptscan import standardize_result_dict

def test_standardize_result_dict_fallback_on_empty_string():
    """
    Test that standardize_result_dict correctly falls back to other aliases
    if a primary alias exists but contains an empty string.
    """
    # 'path' is primary, 'File Path' is an alias.
    # In the current buggy implementation, if 'path' is "", it returns "",
    # instead of checking 'File Path'.
    item = {
        "path": "",
        "File Path": "actual/path/to/file.py",
        "Local Threat": "80%"
    }

    result = standardize_result_dict(item)

    # This is expected to FAIL before the fix
    assert result["path"] == "actual/path/to/file.py"
    assert result["own_conf"] == "80%"

def test_standardize_result_dict_fallback_on_none():
    """
    Test that it also handles None correctly.
    """
    item = {
        "path": None,
        "File Path": "actual/path/to/file.py"
    }

    result = standardize_result_dict(item)

    # This currently passes because of 'if item[alt] is not None'
    assert result["path"] == "actual/path/to/file.py"

def test_standardize_result_dict_all_empty():
    """
    Test when all matching keys are empty.
    """
    item = {
        "path": "",
        "File Path": ""
    }

    result = standardize_result_dict(item)
    assert result["path"] == ""
