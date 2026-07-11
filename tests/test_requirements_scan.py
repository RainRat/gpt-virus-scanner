import pytest
from gptscan import unpack_content

def test_requirements_txt_unpacking():
    """Test that requirements.txt content is correctly yielded as individual dependency snippets."""
    content = b"flask==2.0.1\nrequests>=2.25.1\n# A comment\n  \ngit+https://github.com/unsafe/repo.git\n"
    name = "requirements.txt"

    snippets = list(unpack_content(name, content))

    # We expect 3 snippets (flask, requests, and the git URL), comments and empty lines should be ignored
    assert len(snippets) == 3

    assert snippets[0][0] == "requirements.txt [Dependency]"
    assert snippets[0][1] == b"flask==2.0.1"

    assert snippets[1][0] == "requirements.txt [Dependency]"
    assert snippets[1][1] == b"requests>=2.25.1"

    assert snippets[2][0] == "requirements.txt [Dependency]"
    assert snippets[2][1] == b"git+https://github.com/unsafe/repo.git"

def test_requirements_txt_in_subdir():
    """Test that requirements.txt in a subdirectory is correctly identified."""
    content = b"numpy"
    name = "subdir/requirements.txt"

    snippets = list(unpack_content(name, content))

    assert len(snippets) == 1
    assert snippets[0][0] == "subdir/requirements.txt [Dependency]"
    assert snippets[0][1] == b"numpy"

def test_requirements_txt_variants():
    """Test that dev-requirements.txt and other variants are also scanned."""
    content = b"pytest"
    name = "dev-requirements.txt"

    snippets = list(unpack_content(name, content))

    assert len(snippets) == 1
    assert snippets[0][0] == "dev-requirements.txt [Dependency]"
    assert snippets[0][1] == b"pytest"
