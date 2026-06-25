import pytest
from gptscan import unpack_content

def test_pyproject_inline_table_triple_quotes():
    """Verify that triple-quoted strings inside inline tables in pyproject.toml are handled."""
    content = b"""
[tool.pdm.scripts]
test = { shell = \"\"\"echo "triple"\"\"\" }
test2 = { cmd = '''echo 'triple2' ''' }
"""
    results = list(unpack_content("pyproject.toml", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert "pyproject.toml [Script: test]" in scripts
    assert 'echo "triple"' in scripts["pyproject.toml [Script: test]"]
    assert "pyproject.toml [Script: test2]" in scripts
    assert "echo 'triple2'" in scripts["pyproject.toml [Script: test2]"]
