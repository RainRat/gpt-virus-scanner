import pytest
from gptscan import unpack_content, Config

def test_dockerfile_extraction():
    content = b"""
FROM python:3.9
RUN pip install malicious-pkg
COPY . /app
CMD ["python", "app.py"]
ENTRYPOINT ["/entrypoint.sh"]
"""
    results = list(unpack_content("Dockerfile", content))

    # Expected instructions: RUN, CMD, ENTRYPOINT
    assert len(results) == 3
    assert results[0] == ("Dockerfile [Instruction 1]", b"pip install malicious-pkg")
    assert results[1] == ("Dockerfile [Instruction 2]", b'["python", "app.py"]')
    assert results[2] == ("Dockerfile [Instruction 3]", b'["/entrypoint.sh"]')

def test_dockerfile_multiline():
    content = b"""
RUN apt-get update && \\
    apt-get install -y \\
    curl \\
    wget
"""
    results = list(unpack_content("Dockerfile", content))
    assert len(results) == 1
    assert results[0][1] == b"apt-get update && apt-get install -y curl wget"

def test_makefile_extraction():
    content = b"""
all: build

build:
\t./build.sh
\t@echo "Done"

clean:
\trm -rf dist
"""
    results = list(unpack_content("Makefile", content))

    # Expected recipes
    assert len(results) == 3
    assert results[0] == ("Makefile [Recipe 1]", b"./build.sh")
    assert results[1] == ("Makefile [Recipe 2]", b'@echo "Done"')
    assert results[2] == ("Makefile [Recipe 3]", b"rm -rf dist")

def test_dockerfile_case_insensitive():
    content = b"RUN echo 'hello'"
    results = list(unpack_content("dockerfile", content))
    assert len(results) == 1
    assert results[0] == ("dockerfile [Instruction 1]", b"echo 'hello'")

def test_makefile_case_insensitive():
    content = b"\techo 'world'"
    results = list(unpack_content("makefile", content))
    assert len(results) == 1
    assert results[0] == ("makefile [Recipe 1]", b"echo 'world'")

def test_stricter_matching():
    # Files that should NOT be treated as containers
    content = b"some content"

    # test_makefile.py should NOT be a container, but it IS a supported script (if .py is in extensions)
    # So it might still yield a result, but not as a [Recipe X] or [Instruction X]
    results = list(unpack_content("test_makefile.py", content))
    for name, _ in results:
        assert "[Recipe" not in name
        assert "[Instruction" not in name

    results = list(unpack_content("my_dockerfile.txt", content))
    for name, _ in results:
        assert "[Instruction" not in name

def test_dockerfile_extension():
    content = b"RUN malicious"
    results = list(unpack_content("prod.dockerfile", content))
    assert len(results) == 1
    assert results[0][0] == "prod.dockerfile [Instruction 1]"

def test_makefile_extension():
    content = b"\tmalicious"
    results = list(unpack_content("module.makefile", content))
    assert len(results) == 1
    assert results[0][0] == "module.makefile [Recipe 1]"

def test_fallback_still_works():
    # If a Dockerfile has no RUN/CMD/ENTRYPOINT, it should still be yielded as a whole file if it matches shebang or something (though Dockerfiles usually don't)
    # More importantly, if a .py file was incorrectly matched as a container, it should fallback to being a .py file.

    content = b"#!/usr/bin/env python\nprint('hello')"
    # "Dockerfile" named file with python content and no Docker instructions
    results = list(unpack_content("Dockerfile", content))
    # It should yield the whole file because of the shebang fallback in unpack_content (step 9)
    assert len(results) == 1
    assert results[0] == ("Dockerfile", content)

def test_dockerfile_interrupted_instruction():
    content = b"""
RUN echo first \\
RUN echo second
"""
    results = list(unpack_content("Dockerfile", content))
    assert len(results) == 2
    assert results[0][1] == b"echo first"
    assert results[1][1] == b"echo second"

def test_dockerfile_with_comments_and_newlines():
    content = b"""
RUN echo part1 \\
    # some comment
    part2 \\

    part3
"""
    results = list(unpack_content("Dockerfile", content))
    assert len(results) == 1
    assert results[0][1] == b"echo part1 part2 part3"
