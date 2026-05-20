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

def test_pyproject_multiline_with_trailing_comment():
    """Verify that triple-quoted strings with trailing comments are parsed correctly."""
    content = b"""
[tool.pdm.scripts]
test = \"\"\"echo hello\"\"\" # trailing comment
next = "echo next"
"""
    results = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert scripts["pyproject.toml [Script: test]"] == "echo hello"
    assert scripts["pyproject.toml [Script: next]"] == "echo next"

def test_pyproject_multiline_block_with_trailing_comment():
    """Verify multiline block with closing quotes and comment on separate line."""
    content = b"""
[tool.pdm.scripts]
test = '''
echo line 1
echo line 2
''' # comment here
next = "echo next"
"""
    results = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert "echo line 1" in scripts["pyproject.toml [Script: test]"]
    assert "echo line 2" in scripts["pyproject.toml [Script: test]"]
    assert "comment here" not in scripts["pyproject.toml [Script: test]"]
    assert scripts["pyproject.toml [Script: next]"] == "echo next"

def test_pyproject_multiline_array():
    """Verify that multi-line arrays in pyproject.toml are handled."""
    content = b"""
[tool.pdm.scripts]
multiline_array = [
    "echo 1",
    "echo 2"
]
"""
    results = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert "pyproject.toml [Script: multiline_array (1)]" in scripts
    assert scripts["pyproject.toml [Script: multiline_array (1)]"] == "echo 1"
    assert "pyproject.toml [Script: multiline_array (2)]" in scripts
    assert scripts["pyproject.toml [Script: multiline_array (2)]"] == "echo 2"

def test_pyproject_array_with_escaped_quotes():
    """Verify that arrays with escaped quotes in pyproject.toml are handled."""
    content = b"""
[tool.pdm.scripts]
escaped = ["echo \\"hello\\"", 'echo \\'world\\'']
"""
    results = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert "pyproject.toml [Script: escaped (1)]" in scripts
    assert scripts["pyproject.toml [Script: escaped (1)]"] == 'echo \\"hello\\"'
    assert "pyproject.toml [Script: escaped (2)]" in scripts
    assert scripts["pyproject.toml [Script: escaped (2)]"] == "echo \\'world\\'"

def test_pyproject_multiline_array_with_comments():
    """Verify that multi-line arrays with comments in pyproject.toml are handled."""
    content = b"""
[tool.pdm.scripts]
multiline_comments = [
    "echo 1", # comment 1
    "echo 2"  # comment 2
] # final comment
"""
    results = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert "pyproject.toml [Script: multiline_comments (1)]" in scripts
    assert scripts["pyproject.toml [Script: multiline_comments (1)]"] == "echo 1"
    assert "pyproject.toml [Script: multiline_comments (2)]" in scripts
    assert scripts["pyproject.toml [Script: multiline_comments (2)]"] == "echo 2"
