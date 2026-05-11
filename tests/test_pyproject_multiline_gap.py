import pytest
from gptscan import unpack_content

def test_pyproject_multiline_script_gap():
    """Verify that multi-line scripts (triple quotes) in pyproject.toml are handled."""
    content = b"""
[tool.pdm.scripts]
multiline = '''
echo "line 1"
echo "line 2"
'''
"""
    results = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert "pyproject.toml [Script: multiline]" in scripts
    assert 'echo "line 1"' in scripts["pyproject.toml [Script: multiline]"]
    assert 'echo "line 2"' in scripts["pyproject.toml [Script: multiline]"]

def test_pyproject_inline_comment_gap():
    """Verify that inline comments in pyproject.toml script values are stripped."""
    content = b"""
[tool.pdm.scripts]
with-comment = "echo hello" # this is a comment
"""
    results = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert "pyproject.toml [Script: with-comment]" in scripts
    assert scripts["pyproject.toml [Script: with-comment]"] == "echo hello"

def test_pyproject_hash_in_string():
    """Verify that a hash character inside a string is NOT stripped."""
    content = b"""
[tool.pdm.scripts]
hash-in-str = "echo #not-a-comment"
"""
    results = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert "pyproject.toml [Script: hash-in-str]" in scripts
    assert scripts["pyproject.toml [Script: hash-in-str]"] == "echo #not-a-comment"
