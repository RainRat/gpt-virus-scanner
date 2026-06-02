import pytest
from gptscan import unpack_content

def test_makefile_shell_assignment():
    """Verify that Makefile shell assignment operator (!=) is correctly extracted."""
    content = b"""
SHELL_VAR != echo "malicious"
NORMAL_VAR = normal
"""
    results = list(unpack_content("Makefile", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert "Makefile [Variable 1]" in scripts
    assert scripts["Makefile [Variable 1]"] == 'echo "malicious"'

    assert "Makefile [Variable 2]" in scripts
    assert scripts["Makefile [Variable 2]"] == 'normal'
