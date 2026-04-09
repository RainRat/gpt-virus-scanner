import pytest
from gptscan import unpack_content

def test_extract_github_actions_run():
    """Test extraction from a GitHub Actions workflow (multi-line run)."""
    yaml_content = b"""
name: CI
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Build
        run: |
          npm install
          npm run build
      - name: Test
        run: npm test
"""
    results = list(unpack_content("workflow.yml", yaml_content))

    assert len(results) == 2
    assert results[0][0] == "workflow.yml [Script 1]"
    assert b"npm install" in results[0][1]
    assert b"npm run build" in results[0][1]
    assert results[1][0] == "workflow.yml [Script 2]"
    assert results[1][1] == b"npm test"

def test_extract_gitlab_ci_script():
    """Test extraction from a GitLab CI file (list-based script)."""
    yaml_content = b"""
build-job:
  stage: build
  script:
    - echo "Compiling the code..."
    - make
  after_script:
    - echo "Cleaning up..."
"""
    results = list(unpack_content(".gitlab-ci.yml", yaml_content))

    assert len(results) == 2
    assert results[0][0] == ".gitlab-ci.yml [Script 1]"
    assert b"echo \"Compiling the code...\"" in results[0][1]
    assert b"make" in results[0][1]
    assert results[1][0] == ".gitlab-ci.yml [Script 2]"
    assert results[1][1] == b"echo \"Cleaning up...\""

def test_extract_generic_yaml_command():
    """Test extraction from a generic YAML file (single line command)."""
    yaml_content = b"""
task:
  name: deploy
  command: ./deploy.sh --env production
"""
    results = list(unpack_content("task.yaml", yaml_content))

    assert len(results) == 1
    assert results[0][0] == "task.yaml [Script 1]"
    assert results[0][1] == b"./deploy.sh --env production"

def test_extract_dash_prefixed_key():
    """Test extraction when the key is prefixed by a dash (e.g., - run: echo 1)."""
    yaml_content = b"""
steps:
  - run: echo "dash prefixed"
"""
    results = list(unpack_content("dash.yml", yaml_content))
    assert len(results) == 1
    assert results[0][1] == b"echo \"dash prefixed\""

def test_yaml_fallback_if_no_scripts():
    """Test that it falls back to yielding the whole file if no script keys are found."""
    yaml_content = b"""
metadata:
  name: my-app
  version: 1.0.0
"""
    results = list(unpack_content("meta.yaml", yaml_content))

    # Should yield the whole file as a single snippet
    assert len(results) == 1
    assert results[0][0] == "meta.yaml"
    assert results[0][1] == yaml_content
