import pytest
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
    assert "pyproject.toml [Script: composite]" in names
    assert "pyproject.toml [Script: hatch-test]" in names
    assert "pyproject.toml [Script: ignored]" not in names

    # Verify content
    scripts = {s[0]: s[1].decode() for s in snippets}
    assert scripts["pyproject.toml [Script: spam-cli]"] == "spam:main_cli"
    assert scripts["pyproject.toml [Script: pdm-task]"] == "echo pdm"

def test_is_container_pyproject():
    assert Config.is_container("pyproject.toml") is True
    assert Config.is_container("path/to/pyproject.toml") is True
