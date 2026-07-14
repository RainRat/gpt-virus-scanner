import pytest
from gptscan import unpack_content, Config

def test_unpack_requirements_txt_basic():
    """Test that unpack_content correctly extracts dependencies from requirements.txt."""
    content = b"""
flask==2.0.1
requests>=2.25.1
# A comment line
numpy

  # Another comment with spaces
pandas
"""
    results = list(unpack_content("requirements.txt", content))

    # Should find 4 dependencies
    assert len(results) == 4

    names = [r[0] for r in results]
    assert "requirements.txt [Dependency: flask==2.0.1]" in names
    assert "requirements.txt [Dependency: requests>=2.25.1]" in names
    assert "requirements.txt [Dependency: numpy]" in names
    assert "requirements.txt [Dependency: pandas]" in names

    contents = [r[1] for r in results]
    assert b"flask==2.0.1" in contents
    assert b"requests>=2.25.1" in contents
    assert b"numpy" in contents
    assert b"pandas" in contents

def test_unpack_requirements_variants():
    """Test that variants of requirements.txt are also handled."""
    content = b"django\npytest"

    # dev-requirements.txt
    results = list(unpack_content("dev-requirements.txt", content))
    assert len(results) == 2
    assert "dev-requirements.txt [Dependency: django]" in results[0][0]

    # requirements.in
    results = list(unpack_content("requirements.in", content))
    assert len(results) == 2
    assert "requirements.in [Dependency: django]" in results[0][0]

    # prod.requirements.txt
    results = list(unpack_content("prod.requirements.txt", content))
    assert len(results) == 2
    assert "prod.requirements.txt [Dependency: django]" in results[0][0]

def test_unpack_requirements_with_markers():
    """Test requirements with environment markers."""
    content = b"pkg-a; python_version < '3.7'\npkg-b == 1.0; sys_platform == 'win32'"
    results = list(unpack_content("requirements.txt", content))

    assert len(results) == 2
    assert b"pkg-a; python_version < '3.7'" in [r[1] for r in results]
    assert b"pkg-b == 1.0; sys_platform == 'win32'" in [r[1] for r in results]

def test_unpack_requirements_with_options():
    """Test requirements with pip options like --index-url."""
    content = b"--index-url https://pypi.org/simple\nflask\n-r other.txt"
    results = list(unpack_content("requirements.txt", content))

    # We treat options as dependencies for scanning purposes (suspicious URLs in index-url are worth scanning)
    assert len(results) == 3
    assert b"--index-url https://pypi.org/simple" in [r[1] for r in results]
    assert b"-r other.txt" in [r[1] for r in results]

def test_is_container_requirements():
    """Verify that requirements files are recognized as containers."""
    assert Config.is_container("requirements.txt") is True
    assert Config.is_container("requirements.in") is True
    assert Config.is_container("dev-requirements.txt") is True
    assert Config.is_container("path/to/prod-requirements.txt") is True
    assert Config.is_container("not-requirements.txt") is True # Because of .endswith('requirements.txt')

    # Should not be container if it's a member
    assert Config.is_supported_file("requirements.txt", is_member=True) is False
