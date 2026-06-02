import pytest
from gptscan import unpack_content

def test_pyproject_inline_table_with_bracket_in_string():
    """Verify bug where bracket in string breaks inline table array parsing."""
    content = b"""
[tool.pdm.scripts]
test = { cmd = ["echo", "bracket ] here"] }
"""
    results = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert "pyproject.toml [Script: test (1)]" in scripts
    assert scripts["pyproject.toml [Script: test (1)]"] == "echo"
    assert "pyproject.toml [Script: test (2)]" in scripts
    assert scripts["pyproject.toml [Script: test (2)]"] == "bracket ] here"

def test_pyproject_multiline_array_with_bracket_in_string():
    """Verify bug where bracket in string breaks multiline array parsing."""
    content = b"""
[tool.pdm.scripts]
test = [
    "echo ]",
    "done"
]
"""
    results = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert "pyproject.toml [Script: test (1)]" in scripts
    assert scripts["pyproject.toml [Script: test (1)]"] == "echo ]"
    assert "pyproject.toml [Script: test (2)]" in scripts
    assert scripts["pyproject.toml [Script: test (2)]"] == "done"

def test_pyproject_multiline_array_stops_too_early():
    """Verify that multiline array parsing doesn't stop at a bracket inside a string."""
    content = b"""
[tool.pdm.scripts]
test = [
    "echo ]"
]
"""
    results = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert "pyproject.toml [Script: test (1)]" in scripts
    assert scripts["pyproject.toml [Script: test (1)]"] == "echo ]"

def test_pyproject_inline_table_with_escaped_quotes_and_brackets():
    """Verify robust parsing of inline tables with complex strings."""
    content = b"""
[tool.pdm.scripts]
test = { cmd = ["echo \\"escaped ] quote\\"", 'single ] quote'] }
"""
    results = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert scripts["pyproject.toml [Script: test (1)]"] == 'echo \\"escaped ] quote\\"'
    assert scripts["pyproject.toml [Script: test (2)]"] == 'single ] quote'
