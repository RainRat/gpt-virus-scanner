import pytest
from gptscan import unpack_content

def test_dockerfile_onbuild_extraction():
    """Verify that ONBUILD RUN/CMD/ENTRYPOINT instructions are extracted."""
    content = b"""
FROM alpine
ONBUILD RUN echo "onbuild run"
ONBUILD CMD ["echo", "onbuild cmd"]
ONBUILD ENTRYPOINT ["echo", "onbuild entrypoint"]
"""
    results = list(unpack_content("Dockerfile", content))
    # Expected: ONBUILD RUN, ONBUILD CMD, ONBUILD ENTRYPOINT
    assert len(results) == 3
    assert results[0][1] == b'echo "onbuild run"'
    assert results[1][1] == b'["echo", "onbuild cmd"]'
    assert results[2][1] == b'["echo", "onbuild entrypoint"]'

def test_dockerfile_healthcheck_extraction():
    """Verify that HEALTHCHECK CMD instructions are extracted."""
    content = b"""
FROM nginx
HEALTHCHECK --interval=5m --timeout=3s \\
  CMD curl -f http://localhost/ || exit 1
"""
    results = list(unpack_content("Dockerfile", content))
    assert len(results) == 1
    assert b"curl -f http://localhost/ || exit 1" in results[0][1]

def test_dockerfile_mixed_instructions():
    """Verify a mix of regular and prefixed instructions."""
    content = b"""
RUN echo "regular"
ONBUILD RUN echo "onbuild"
HEALTHCHECK CMD echo "health"
"""
    results = list(unpack_content("Dockerfile", content))
    assert len(results) == 3
    contents = [r[1] for r in results]
    assert b'echo "regular"' in contents
    assert b'echo "onbuild"' in contents
    assert b'echo "health"' in contents
