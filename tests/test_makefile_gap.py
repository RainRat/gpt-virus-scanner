import pytest
from gptscan import unpack_content

def test_makefile_variable_extraction():
    """Verify that Makefile variable assignments are correctly extracted."""
    content = b"""
MALICIOUS_VAR = rm -rf /
IMMEDIATE := echo "immediate"
APPEND += --flag
DEFAULT ?= default_val
EMPTY =
all:
\techo "done"
"""
    results = list(unpack_content("Makefile", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert "Makefile [Variable 1]" in scripts
    assert scripts["Makefile [Variable 1]"] == "rm -rf /"

    assert "Makefile [Variable 2]" in scripts
    assert scripts["Makefile [Variable 2]"] == 'echo "immediate"'

    assert "Makefile [Variable 3]" in scripts
    assert scripts["Makefile [Variable 3]"] == "--flag"

    assert "Makefile [Variable 4]" in scripts
    assert scripts["Makefile [Variable 4]"] == "default_val"

    # EMPTY variable should not be yielded if val.strip() is empty
    assert "Makefile [Variable 5]" not in scripts

    assert "Makefile [Recipe 1]" in scripts
    assert scripts["Makefile [Recipe 1]"] == 'echo "done"'

def test_makefile_multiline_variable():
    """Verify that multi-line Makefile variable assignments are correctly extracted."""
    content = b"""
MULTILINE = echo "part 1" \\
            "part 2" \\
            "part 3"
NEXT = after
"""
    results = list(unpack_content("Makefile", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert "Makefile [Variable 1]" in scripts
    assert scripts["Makefile [Variable 1]"] == 'echo "part 1" "part 2" "part 3"'

    assert "Makefile [Variable 2]" in scripts
    assert scripts["Makefile [Variable 2]"] == "after"

def test_makefile_mixed_variables_and_recipes():
    """Verify mixed variables and recipes are extracted in order."""
    content = b"""
VAR1 = val1
target1:
\tcmd1
VAR2 = val2
target2:
\tcmd2
"""
    results = list(unpack_content("Makefile", content))
    # Order should be preserved in numbering
    assert results[0][0] == "Makefile [Variable 1]"
    assert results[0][1] == b"val1"
    assert results[1][0] == "Makefile [Recipe 1]"
    assert results[1][1] == b"cmd1"
    assert results[2][0] == "Makefile [Variable 2]"
    assert results[2][1] == b"val2"
    assert results[3][0] == "Makefile [Recipe 2]"
    assert results[3][1] == b"cmd2"
