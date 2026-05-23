import pytest
from gptscan import unpack_content

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
