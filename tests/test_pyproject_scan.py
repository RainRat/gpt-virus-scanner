import pytest
from gptscan import unpack_content

def test_unpack_pyproject_toml_scripts():
    content = b"""
[project]
name = "test-project"

[project.scripts]
my-cmd = "my_module:main"
another = 'run_this --flag'

[tool.poetry.scripts]
poetry-cmd = "poetry_mod:app"

[tool.poe.tasks]
task1 = "echo hello"
task2 = { cmd = "ls -la", help = "list files" }
task3 = { shell = "rm -rf /" }

[tool.taskipy.tasks]
test = "pytest"
"""
    results = list(unpack_content("pyproject.toml", content))

    # Check if all scripts and tasks were extracted
    names = [r[0] for r in results]
    assert "pyproject.toml [Script: my-cmd]" in names
    assert "pyproject.toml [Script: another]" in names
    assert "pyproject.toml [Script: poetry-cmd]" in names
    assert "pyproject.toml [Task: task1]" in names
    assert "pyproject.toml [Task: task2]" in names
    assert "pyproject.toml [Task: task3]" in names
    assert "pyproject.toml [Task: test]" in names

    # Check extracted values
    script_vals = {r[0]: r[1].decode() for r in results}
    assert script_vals["pyproject.toml [Script: my-cmd]"] == "my_module:main"
    assert script_vals["pyproject.toml [Script: another]"] == "run_this --flag"
    assert script_vals["pyproject.toml [Task: task2]"] == "ls -la"
    assert script_vals["pyproject.toml [Task: task3]"] == "rm -rf /"

def test_unpack_pyproject_toml_no_scripts():
    content = b"""
[project]
name = "test-project"
version = "0.1.0"
"""
    # Should yield nothing if no scripts/tasks are found
    results = list(unpack_content("pyproject.toml", content))
    assert len(results) == 0

def test_unpack_pyproject_toml_robustness():
    content = b"""
[project.scripts]
# A comment
valid = "cmd"
# Another comment
[tool.poetry.scripts]
pvalid = "pcmd"
"""
    results = list(unpack_content("pyproject.toml", content))
    assert len(results) == 2
    names = [r[0] for r in results]
    assert "pyproject.toml [Script: valid]" in names
    assert "pyproject.toml [Script: pvalid]" in names
