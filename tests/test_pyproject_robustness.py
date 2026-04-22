import pytest
from gptscan import unpack_content

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

    assert "pyproject.toml [Script: multi]" in scripts
    assert scripts["pyproject.toml [Script: multi]"] == "lint\", \"test" # Due to simple regex stripping of []

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

def test_pyproject_mixed_scripts():
    content = b"""
[project.scripts]
cli = "pkg:main"

[tool.pdm.scripts]
pdm-flat = "echo flat"
pdm-table = { shell = "echo table" }

[tool.pdm.scripts.pdm-nested]
cmd = "echo nested"
"""
    snippets = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in snippets}

    assert "pyproject.toml [Script: cli]" in scripts
    assert scripts["pyproject.toml [Script: cli]"] == "pkg:main"
    assert "pyproject.toml [Script: pdm-flat]" in scripts
    assert scripts["pyproject.toml [Script: pdm-flat]"] == "echo flat"
    assert "pyproject.toml [Script: pdm-table]" in scripts
    assert scripts["pyproject.toml [Script: pdm-table]"] == "echo table"
    assert "pyproject.toml [Script: pdm-nested]" in scripts
    assert scripts["pyproject.toml [Script: pdm-nested]"] == "echo nested"

def test_pyproject_case_insensitivity():
    content = b"""
[TOOL.pdm.SCRIPTS.UPPER]
CMD = "echo upper"
"""
    snippets = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in snippets}

    assert "pyproject.toml [Script: UPPER]" in scripts
    assert scripts["pyproject.toml [Script: UPPER]"] == "echo upper"
