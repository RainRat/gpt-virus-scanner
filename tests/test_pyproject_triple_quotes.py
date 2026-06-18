import pytest
from gptscan import unpack_content

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

    # If it fails, these assertions will fail
    assert "pyproject.toml [Script: triple_array (1)]" in scripts
    assert 'echo "triple 1"' in scripts["pyproject.toml [Script: triple_array (1)]"]
    assert "pyproject.toml [Script: triple_array (2)]" in scripts
    assert 'echo "triple 2"' in scripts["pyproject.toml [Script: triple_array (2)]"]
