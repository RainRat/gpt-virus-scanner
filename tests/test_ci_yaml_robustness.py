import pytest
from gptscan import unpack_content

def test_yaml_list_with_comments():
    content = b"""
script:
  - echo "1"
  # comment
  - echo "2"
"""
    snippets = list(unpack_content("test.yml", content))
    commands = [s[1].decode('utf-8') for s in snippets]
    assert 'echo "1"\necho "2"' in commands

def test_yaml_list_with_empty_lines():
    content = b"""
script:
  - echo "1"

  - echo "2"
"""
    snippets = list(unpack_content("test.yml", content))
    commands = [s[1].decode('utf-8') for s in snippets]
    assert 'echo "1"\necho "2"' in commands

def test_yaml_list_indented_comments():
    content = b"""
script:
  - echo "1"
    # comment same level as script
  - echo "2"
"""
    snippets = list(unpack_content("test.yml", content))
    commands = [s[1].decode('utf-8') for s in snippets]
    assert 'echo "1"\necho "2"' in commands

def test_yaml_complex_list():
    content = b"""
before_script:
  - export VAR=1
  # Setup steps
  - ./setup.sh

script:
  - make build
  # Run tests
  - make test
"""
    snippets = list(unpack_content(".gitlab-ci.yml", content))
    assert len(snippets) == 2
    assert b"export VAR=1\n./setup.sh" in snippets[0][1]
    assert b"make build\nmake test" in snippets[1][1]
