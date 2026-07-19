import os
from pathlib import Path
import pytest
from gptscan import get_env_file_paths, unpack_content, Config

def test_get_env_file_paths(tmp_path, monkeypatch):
    # Mock home directory
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)

    # Mock current directory
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    # Create some .env files
    env1 = home / ".env"
    env1.write_text("KEY1=VALUE1")

    env2 = cwd / ".env.local"
    env2.write_text("KEY2=VALUE2")

    env3 = cwd / ".env.test"
    env3.write_text("KEY3=VALUE3")

    # Non-env file
    (cwd / "test.txt").write_text("test")

    paths = get_env_file_paths()

    assert str(env1.absolute()) in paths
    assert str(env2.absolute()) in paths
    assert str(env3.absolute()) in paths
    assert len(paths) == 3

def test_unpack_env_content():
    content = b"""
KEY1=VALUE1
# Comment
export KEY2=VALUE2
  KEY3 = VALUE3
"""
    name = ".env"
    snippets = list(unpack_content(name, content))

    assert len(snippets) == 3
    assert snippets[0] == (".env [Env: KEY1]", b"KEY1=VALUE1")
    assert snippets[1] == (".env [Env: KEY2]", b"export KEY2=VALUE2")
    assert snippets[2] == (".env [Env: KEY3]", b"KEY3 = VALUE3")

def test_is_container_env():
    assert Config.is_container(".env") is True
    assert Config.is_container(".env.local") is True
    assert Config.is_container(".env.development.local") is True
    assert Config.is_container("env.txt") is False
