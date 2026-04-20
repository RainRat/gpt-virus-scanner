import pytest
from gptscan import unpack_content, Config

def test_unpack_pyproject_scripts():
    """Test that unpack_content correctly extracts scripts from pyproject.toml."""
    content = b"""
[project]
name = "test-project"

[project.scripts]
run-app = "myapp.main:run"
test = "pytest"

[tool.poetry.scripts]
poetry-run = "poetry_app.main:start"
"""
    results = list(unpack_content("pyproject.toml", content))

    # Should find 3 scripts
    assert len(results) == 3

    names = [r[0] for r in results]
    assert "pyproject.toml [Script: run-app]" in names
    assert "pyproject.toml [Script: test]" in names
    assert "pyproject.toml [Poetry Script: poetry-run]" in names

    script_contents = [r[1] for r in results]
    assert b"myapp.main:run" in script_contents
    assert b"pytest" in script_contents
    assert b"poetry_app.main:start" in script_contents

def test_unpack_pyproject_tasks_inline_tables():
    """Test extraction of tasks from poe and taskipy with inline tables."""
    content = b"""
[tool.poe.tasks]
clean = "rm -rf dist"
build = { cmd = "python -m build", help = "Build the package" }
test = { shell = "pytest --cov", help = "Run tests" }

[tool.taskipy.tasks]
lint = "flake8"
format = { cmd = "black .", help = "Format code" }
"""
    results = list(unpack_content("pyproject.toml", content))

    # 3 from poe, 2 from taskipy
    assert len(results) == 5

    names = [r[0] for r in results]
    assert "pyproject.toml [Poe Task: clean]" in names
    assert "pyproject.toml [Poe Task: build]" in names
    assert "pyproject.toml [Poe Task: test]" in names
    assert "pyproject.toml [Taskipy Task: lint]" in names
    assert "pyproject.toml [Taskipy Task: format]" in names

    script_contents = [r[1] for r in results]
    assert b"rm -rf dist" in script_contents
    assert b"python -m build" in script_contents
    assert b"pytest --cov" in script_contents
    assert b"flake8" in script_contents
    assert b"black ." in script_contents

def test_pyproject_parsing_robustness():
    """Test parsing logic with comments and different section orders."""
    content = b"""
# Top level comment
[project.scripts]
# Script comment
start = "python main.py" # inline comment

[tool.something-else]
ignored = "value"

[tool.poe.tasks]
task1 = "echo 1"
# middle comment
task2 = "echo 2"
"""
    results = list(unpack_content("pyproject.toml", content))

    assert len(results) == 3
    names = [r[0] for r in results]
    assert "pyproject.toml [Script: start]" in names
    assert "pyproject.toml [Poe Task: task1]" in names
    assert "pyproject.toml [Poe Task: task2]" in names

def test_pyproject_no_scripts():
    """Test that pyproject.toml with no relevant sections yields nothing."""
    content = b"""
[project]
name = "test"
version = "0.1.0"

[tool.poetry]
description = "No scripts here"
"""
    results = list(unpack_content("pyproject.toml", content))
    assert len(results) == 0

def test_pyproject_malformed_section():
    """Test behavior with malformed or empty sections."""
    content = b"""
[project.scripts]
# Empty
[tool.poe.tasks]
valid = "command"
invalid
"""
    results = list(unpack_content("pyproject.toml", content))
    assert len(results) == 1
    assert results[0][0] == "pyproject.toml [Poe Task: valid]"
