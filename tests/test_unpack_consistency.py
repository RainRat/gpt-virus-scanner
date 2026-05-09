import pytest
from gptscan import unpack_content, Config

def test_repro_dockerfile_hyphen():
    content = b"RUN echo 'hello'"
    name = "my-dockerfile"
    assert Config.is_container(name) is True

    results = list(unpack_content(name, content))
    assert len(results) == 1
    # If it was unpacked as a Dockerfile, it should have [Instruction 1]
    assert "[Instruction 1]" in results[0][0], f"Expected instruction for {name}, got {results[0][0]}"

def test_repro_makefile_hyphen():
    content = b"\techo 'hello'"
    name = "my-makefile"
    assert Config.is_container(name) is True

    results = list(unpack_content(name, content))
    assert len(results) == 1
    assert "[Recipe 1]" in results[0][0], f"Expected recipe for {name}, got {results[0][0]}"

def test_repro_pyproject_hyphen():
    content = b'[project.scripts]\nspam = "spam:main"'
    name = "my-pyproject.toml"
    assert Config.is_container(name) is True

    results = list(unpack_content(name, content))
    assert len(results) == 1
    assert "[Script: spam]" in results[0][0], f"Expected script for {name}, got {results[0][0]}"
