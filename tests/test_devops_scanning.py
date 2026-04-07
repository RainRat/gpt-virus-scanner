import pytest
from gptscan import unpack_content

def test_unpack_dockerfile_instructions():
    """Test that unpack_content correctly extracts instructions from a Dockerfile."""
    docker_content = b"""
FROM python:3.9
# Install dependencies
RUN apt-get update && \
    apt-get install -y curl
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
ENTRYPOINT ["/usr/bin/entrypoint.sh"]
"""
    results = list(unpack_content("Dockerfile", docker_content))

    # We expect 4 instructions: 2 RUN, 1 CMD, 1 ENTRYPOINT
    assert len(results) == 4

    # Check RUN with line continuation
    assert results[0][0] == "Dockerfile [Instruction 1]"
    assert b"apt-get update &&     apt-get install -y curl" in results[0][1]

    # Check RUN simple
    assert results[1][0] == "Dockerfile [Instruction 2]"
    assert b"pip install -r requirements.txt" in results[1][1]

    # Check CMD (JSON form should be joined)
    assert results[2][0] == "Dockerfile [Instruction 3]"
    assert b"python app.py" in results[2][1]

    # Check ENTRYPOINT (JSON form should be joined)
    assert results[3][0] == "Dockerfile [Instruction 4]"
    assert b"/usr/bin/entrypoint.sh" in results[3][1]

def test_unpack_makefile_recipes():
    """Test that unpack_content correctly extracts recipes from a Makefile."""
    makefile_content = b"""
all: build test

build:
\t@echo "Building..."
\tpython setup.py build

test:
\tpytest tests/

clean:
\trm -rf build/
"""
    results = list(unpack_content("Makefile", makefile_content))

    # We expect 3 recipes: build, test, clean
    assert len(results) == 3

    # Check build recipe
    assert results[0][0] == "Makefile [Recipe 1]"
    assert b"@echo \"Building...\"\npython setup.py build" in results[0][1]

    # Check test recipe
    assert results[1][0] == "Makefile [Recipe 2]"
    assert b"pytest tests/" in results[1][1]

    # Check clean recipe
    assert results[2][0] == "Makefile [Recipe 3]"
    assert b"rm -rf build/" in results[2][1]

def test_dockerfile_case_insensitivity():
    """Test that Dockerfile detection is case-insensitive."""
    docker_content = b"RUN echo 'hello'"
    results = list(unpack_content("dockerfile", docker_content))
    assert len(results) == 1
    assert "dockerfile [Instruction 1]" in results[0][0]

def test_makefile_case_insensitivity():
    """Test that Makefile detection is case-insensitive."""
    makefile_content = b"target:\n\techo 'hi'"
    results = list(unpack_content("makefile", makefile_content))
    assert len(results) == 1
    assert "makefile [Recipe 1]" in results[0][0]
