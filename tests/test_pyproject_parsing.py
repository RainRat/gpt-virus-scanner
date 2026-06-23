import pytest
import re
from gptscan import unpack_content, Config

def test_pyproject_toml_extraction():
    content = b"""
[project.scripts]
spam-cli = "spam:main_cli"

[tool.poetry.scripts]
poetry-run = "module.sub:func"

[tool.pdm.scripts]
pdm-task = "echo pdm"
composite = { composite = ["pdm-task", "ls"] }

[tool.hatch.scripts]
hatch-test = "pytest"

[other.section]
ignored = "rm -rf /"
"""
    snippets = list(unpack_content("pyproject.toml", content))
    names = [s[0] for s in snippets]

    assert "pyproject.toml [Script: spam-cli]" in names
    assert "pyproject.toml [Script: poetry-run]" in names
    assert "pyproject.toml [Script: pdm-task]" in names
    assert "pyproject.toml [Script: composite (1)]" in names
    assert "pyproject.toml [Script: composite (2)]" in names
    assert "pyproject.toml [Script: hatch-test]" in names
    assert "pyproject.toml [Script: ignored]" not in names

    # Verify content
    scripts = {s[0]: s[1].decode() for s in snippets}
    assert scripts["pyproject.toml [Script: spam-cli]"] == "spam:main_cli"
    assert scripts["pyproject.toml [Script: pdm-task]"] == "echo pdm"

def test_pyproject_gui_scripts_extraction():
    """Verify that [project.gui-scripts] section is correctly extracted."""
    content = b"""
[project.gui-scripts]
spam-gui = "spam:main_gui"
"""
    results = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in results}
    assert "pyproject.toml [Script: spam-gui]" in scripts
    assert scripts["pyproject.toml [Script: spam-gui]"] == "spam:main_gui"

def test_pyproject_quoted_key_extraction():
    """Verify that quoted keys with spaces in pyproject.toml are extracted."""
    content = b"""
[tool.pdm.scripts]
"quoted script" = "echo hello"
'single quoted' = "echo world"
"""
    results = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in results}
    assert "pyproject.toml [Script: quoted script]" in scripts
    assert scripts["pyproject.toml [Script: quoted script]"] == "echo hello"
    assert "pyproject.toml [Script: single quoted]" in scripts
    assert scripts["pyproject.toml [Script: single quoted]"] == "echo world"

def test_pyproject_multiline_script():
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

def test_pyproject_inline_comment():
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

def test_pyproject_complex_multiline_array():
    """Verify multiline arrays with internal and trailing comments."""
    content = b"""
[tool.pdm.scripts]
complex = [
    "echo 1", # internal 1
    "echo 2", # internal 2
] # trailing
next = "echo next"
"""
    results = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert "pyproject.toml [Script: complex (1)]" in scripts
    assert scripts["pyproject.toml [Script: complex (1)]"] == "echo 1"
    assert "pyproject.toml [Script: complex (2)]" in scripts
    assert scripts["pyproject.toml [Script: complex (2)]"] == "echo 2"
    assert "pyproject.toml [Script: next]" in scripts
    assert scripts["pyproject.toml [Script: next]"] == "echo next"

def test_pyproject_nested_sections_mixed():
    """Verify mixed flat and nested sections."""
    content = b"""
[project.scripts]
cli = "pkg:main"

[tool.pdm.scripts]
pdm-flat = "echo flat"

[tool.pdm.scripts.nested]
cmd = "echo nested"

[tool.poe.tasks]
poe-flat = "echo poe"

[tool.poe.tasks.nested-poe]
shell = "echo nested poe"
"""
    results = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert scripts["pyproject.toml [Script: cli]"] == "pkg:main"
    assert scripts["pyproject.toml [Script: pdm-flat]"] == "echo flat"
    assert scripts["pyproject.toml [Script: nested]"] == "echo nested"
    assert scripts["pyproject.toml [Script: poe-flat]"] == "echo poe"
    assert scripts["pyproject.toml [Script: nested-poe]"] == "echo nested poe"

def test_pyproject_multiline_string_single_line():
    """Verify triple quotes on a single line with comments."""
    content = b"""
[tool.pdm.scripts]
test = '''echo hello''' # comment
next = \"\"\"echo world\"\"\" # another
"""
    results = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert scripts["pyproject.toml [Script: test]"] == "echo hello"
    assert scripts["pyproject.toml [Script: next]"] == "echo world"

def test_pyproject_invalid_sections_ignored():
    """Verify sections that are not script-related are ignored."""
    content = b"""
[build-system]
requires = ["setuptools"]

[tool.other]
data = "important"
"""
    results = list(unpack_content("pyproject.toml", content))
    assert len(results) == 0

def test_pyproject_inline_table_script():
    """Verify scripts defined in inline tables."""
    content = b"""
[tool.pdm.scripts]
test = { cmd = "pytest", help = "run tests" }
shell-task = { shell = "echo hi" }
"""
    results = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert scripts["pyproject.toml [Script: test]"] == "pytest"
    assert scripts["pyproject.toml [Script: shell-task]"] == "echo hi"

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

def test_pyproject_pdm_nested_scripts():
    content = b"""
[tool.pdm.scripts.test]
cmd = "pytest tests"
help = "Run tests"

[tool.pdm.scripts.post_install]
shell = "echo done"

[tool.pdm.scripts.multi]
composite = ["lint", "test"]
"""
    snippets = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in snippets}

    assert "pyproject.toml [Script: test]" in scripts
    assert scripts["pyproject.toml [Script: test]"] == "pytest tests"

    assert "pyproject.toml [Script: post_install]" in scripts
    assert scripts["pyproject.toml [Script: post_install]"] == "echo done"

    assert "pyproject.toml [Script: multi (1)]" in scripts
    assert scripts["pyproject.toml [Script: multi (1)]"] == "lint"
    assert "pyproject.toml [Script: multi (2)]" in scripts
    assert scripts["pyproject.toml [Script: multi (2)]"] == "test"

def test_pyproject_hatch_scripts():
    content = b"""
[tool.hatch.scripts]
test = "pytest"
nested = { cmd = "echo hatch" }
"""
    snippets = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in snippets}

    assert "pyproject.toml [Script: test]" in scripts
    assert scripts["pyproject.toml [Script: test]"] == "pytest"

    assert "pyproject.toml [Script: nested]" in scripts
    assert scripts["pyproject.toml [Script: nested]"] == "echo hatch"

def test_pyproject_case_insensitivity():
    content = b"""
[TOOL.pdm.SCRIPTS.UPPER]
CMD = "echo upper"
"""
    snippets = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in snippets}

    assert "pyproject.toml [Script: UPPER]" in scripts
    assert scripts["pyproject.toml [Script: UPPER]"] == "echo upper"

def test_pyproject_header_robustness():
    """Verify that pyproject.toml sections with trailing comments and whitespace are recognized."""
    content = b"""
[tool.pdm.scripts] # comment
test = "echo hello"

[ tool.pdm.scripts.nested ] # another comment
cmd = "pytest"
"""
    results = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert len(results) == 2
    assert "pyproject.toml [Script: test]" in scripts
    assert scripts["pyproject.toml [Script: test]"] == "echo hello"
    assert "pyproject.toml [Script: nested]" in scripts
    assert scripts["pyproject.toml [Script: nested]"] == "pytest"

def test_pyproject_array_with_triple_quotes():
    """Verify that arrays containing triple-quoted strings in pyproject.toml are handled."""
    content = b"""
[tool.pdm.scripts]
triple_array = [
    '''echo "triple 1"''',
    \"\"\"echo "triple 2"\"\"\"
]
"""
    results = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert "pyproject.toml [Script: triple_array (1)]" in scripts
    assert 'echo "triple 1"' in scripts["pyproject.toml [Script: triple_array (1)]"]
    assert "pyproject.toml [Script: triple_array (2)]" in scripts
    assert 'echo "triple 2"' in scripts["pyproject.toml [Script: triple_array (2)]"]

def test_is_container_pyproject():
    assert Config.is_container("pyproject.toml") is True
    assert Config.is_container("path/to/pyproject.toml") is True
