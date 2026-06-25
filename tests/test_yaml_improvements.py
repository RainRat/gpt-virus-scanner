import pytest
from gptscan import unpack_content

def test_yaml_extraction_new_keys():
    """Verify that newly added script keys like 'runcmd', 'bootcmd', 'args', etc. are correctly extracted."""
    yaml_content = b"""
cloud-init:
  runcmd:
    - echo "cloud-init runcmd"
  bootcmd:
    - echo "cloud-init bootcmd"

build:
  setup:
    - apt-get update
  install:
    - pip install .
  test:
    - pytest

meta:
  args:
    - --payload
    - rm -rf /
"""
    results = list(unpack_content("test.yml", yaml_content))
    contents = {r[0]: r[1].decode('utf-8') for r in results}

    assert "test.yml [Script 1]" in contents
    assert 'echo "cloud-init runcmd"' in contents["test.yml [Script 1]"]

    assert "test.yml [Script 2]" in contents
    assert 'echo "cloud-init bootcmd"' in contents["test.yml [Script 2]"]

    assert "test.yml [Script 3]" in contents
    assert "apt-get update" in contents["test.yml [Script 3]"]

    assert "test.yml [Script 4]" in contents
    assert "pip install ." in contents["test.yml [Script 4]"]

    assert "test.yml [Script 5]" in contents
    assert "pytest" in contents["test.yml [Script 5]"]

    assert "test.yml [Script 6]" in contents
    assert "--payload\nrm -rf /" in contents["test.yml [Script 6]"]

def test_yaml_list_same_indentation():
    """Verify that YAML lists at the same indentation level as their key are correctly handled."""
    # This is common in some YAML variants/parsers
    yaml_content = b"""
script:
- echo "item 1"
- echo "item 2"
"""
    results = list(unpack_content("test.yml", yaml_content))
    assert len(results) == 1
    assert "echo \"item 1\"\necho \"item 2\"" in results[0][1].decode('utf-8')

def test_yaml_multiline_list_mixed_improvements():
    """Verify that a mix of single-line and multi-line list items works correctly."""
    yaml_content = b"""
script:
  - echo "start"
  - |
    echo "middle line 1"
    echo "middle line 2"
  - echo "end"
"""
    results = list(unpack_content("test.yml", yaml_content))
    assert len(results) == 1
    text = results[0][1].decode('utf-8')
    assert 'echo "start"' in text
    assert 'echo "middle line 1"' in text
    assert 'echo "middle line 2"' in text
    assert 'echo "end"' in text
