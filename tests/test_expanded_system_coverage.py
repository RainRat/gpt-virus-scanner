import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from gptscan import (
    get_documents_paths,
    get_ruby_gems_paths,
    get_php_composer_paths,
    get_rust_cargo_paths,
    get_go_packages_paths,
    get_system_audit_data
)

def test_get_documents_paths(tmp_path):
    with patch('pathlib.Path.home', return_value=tmp_path):
        docs = tmp_path / "Documents"
        docs.mkdir()
        paths = get_documents_paths()
        assert str(docs) in paths

def test_get_ruby_gems_paths(tmp_path):
    gem_dir = tmp_path / "gems"
    gem_dir.mkdir()
    with patch.dict(os.environ, {"GEM_HOME": str(gem_dir)}):
        paths = get_ruby_gems_paths()
        assert str(gem_dir) in paths

def test_get_php_composer_paths(tmp_path):
    composer_dir = tmp_path / ".composer" / "vendor"
    composer_dir.mkdir(parents=True)
    with patch('pathlib.Path.home', return_value=tmp_path):
        paths = get_php_composer_paths()
        assert str(composer_dir) in paths

def test_get_rust_cargo_paths(tmp_path):
    cargo_dir = tmp_path / ".cargo"
    registry = cargo_dir / "registry"
    git = cargo_dir / "git"
    registry.mkdir(parents=True)
    git.mkdir()
    with patch('pathlib.Path.home', return_value=tmp_path):
        paths = get_rust_cargo_paths()
        assert str(registry) in paths
        assert str(git) in paths

def test_get_go_packages_paths(tmp_path):
    go_dir = tmp_path / "go"
    pkg = go_dir / "pkg"
    pkg.mkdir(parents=True)
    with patch.dict(os.environ, {"GOPATH": str(go_dir)}):
        paths = get_go_packages_paths()
        assert str(pkg) in paths

def test_get_system_audit_data_integration():
    with patch('gptscan.get_ruby_gems_paths', return_value=['/fake/gems']):
        with patch('gptscan.get_php_composer_paths', return_value=['/fake/composer']):
            with patch('gptscan.get_rust_cargo_paths', return_value=['/fake/cargo']):
                with patch('gptscan.get_go_packages_paths', return_value=['/fake/go']):
                    with patch('gptscan.get_documents_paths', return_value=['/fake/docs']):
                        p, _ = get_system_audit_data()
                        assert '/fake/gems' in p
                        assert '/fake/composer' in p
                        assert '/fake/cargo' in p
                        assert '/fake/go' in p
                        assert '/fake/docs' in p
