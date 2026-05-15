import pytest
from gptscan import unpack_content

def test_pyproject_poe_tasks_support():
    """Verify that [tool.poe.tasks] in pyproject.toml is correctly extracted."""
    content = b"""
[tool.poe.tasks]
test = "pytest"
serve = { cmd = "python manage.py runserver" }
expr_task = { expr = "sys.platform" }
script_task = { script = "my_module:main" }

[tool.poe.tasks.deep]
shell = "echo deep"
"""
    snippets = list(unpack_content("pyproject.toml", content))
    names = [s[0] for s in snippets]

    assert "pyproject.toml [Script: test]" in names
    assert "pyproject.toml [Script: serve]" in names
    assert "pyproject.toml [Script: expr_task]" in names
    assert "pyproject.toml [Script: script_task]" in names
    assert "pyproject.toml [Script: deep]" in names

    # Verify content
    scripts = {s[0]: s[1].decode() for s in snippets}
    assert scripts["pyproject.toml [Script: test]"] == "pytest"
    assert scripts["pyproject.toml [Script: serve]"] == "python manage.py runserver"
    assert scripts["pyproject.toml [Script: expr_task]"] == "sys.platform"
    assert scripts["pyproject.toml [Script: script_task]"] == "my_module:main"
    assert scripts["pyproject.toml [Script: deep]"] == "echo deep"
