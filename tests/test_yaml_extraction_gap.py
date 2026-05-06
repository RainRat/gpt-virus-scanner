import pytest
from gptscan import unpack_content

def test_yaml_extraction_additional_keys():
    """Verify that additional script keys like 'commands', 'entry', 'bash', etc. are now correctly extracted."""
    yaml_content = b"""
azure-pipelines:
  steps:
    - bash: |
        echo "Azure bash"
    - pwsh: echo "Azure pwsh"
    - cmd: echo "Azure cmd"
drone-ci:
  steps:
    - commands:
        - echo "Drone command 1"
        - echo "Drone command 2"
pre-commit:
  hooks:
    - entry: python malicious.py
"""
    results = list(unpack_content("test.yml", yaml_content))

    # Check that individual snippets are extracted
    names = [r[0] for r in results]

    assert "test.yml [Script 1]" in names # bash
    assert "test.yml [Script 2]" in names # pwsh
    assert "test.yml [Script 3]" in names # cmd
    assert "test.yml [Script 4]" in names # commands
    assert "test.yml [Script 5]" in names # entry

    contents = {r[0]: r[1].decode('utf-8') for r in results}
    assert 'echo "Azure bash"' in contents["test.yml [Script 1]"]
    assert 'echo "Azure pwsh"' in contents["test.yml [Script 2]"]
    assert 'echo "Azure cmd"' in contents["test.yml [Script 3]"]
    assert 'echo "Drone command 1"\necho "Drone command 2"' in contents["test.yml [Script 4]"]
    assert 'python malicious.py' in contents["test.yml [Script 5]"]

def test_yaml_powershell_key():
    """Verify that the 'powershell' key is also supported."""
    yaml_content = b"""
steps:
  - powershell: Write-Host "Hello"
"""
    results = list(unpack_content("test.yml", yaml_content))
    assert len(results) == 1
    assert results[0][0] == "test.yml [Script 1]"
    assert results[0][1] == b'Write-Host "Hello"'
