import pytest
from gptscan import unpack_content

def test_pyproject_unpacking():
    content = b"""
[project]
name = "test-project"

[project.scripts]
run-app = "python main.py"
test = "pytest"

[tool.poetry.scripts]
poetry-run = "poetry run cmd"

[tool.pdm.scripts]
pdm-task = "pdm run something"

[tool.hatch.scripts]
hatch-script = "hatch run script"

[tool.pixi.tasks]
pixi-task = "pixi run task"

[other.section]
ignored = "should not be yielded"
"""
    snippets = list(unpack_content("pyproject.toml", content))

    expected = [
        ("pyproject.toml [Script: run-app]", b"python main.py"),
        ("pyproject.toml [Script: test]", b"pytest"),
        ("pyproject.toml [Script: poetry-run]", b"poetry run cmd"),
        ("pyproject.toml [Script: pdm-task]", b"pdm run something"),
        ("pyproject.toml [Script: hatch-script]", b"hatch run script"),
        ("pyproject.toml [Script: pixi-task]", b"pixi run task"),
    ]

    assert len(snippets) == len(expected)
    for s in expected:
        assert s in snippets

def test_pyproject_quoting_and_comments():
    content = b"""
[project.scripts]
single = 'single-quoted'
double = "double-quoted"
with-comment = "command" # some comment
"""
    snippets = list(unpack_content("pyproject.toml", content))

    expected = [
        ("pyproject.toml [Script: single]", b"single-quoted"),
        ("pyproject.toml [Script: double]", b"double-quoted"),
        ("pyproject.toml [Script: with-comment]", b"command"),
    ]

    assert len(snippets) == len(expected)
    for s in expected:
        assert s in snippets
