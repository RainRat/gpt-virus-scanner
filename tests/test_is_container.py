import pytest
from pathlib import Path
from gptscan import Config

def test_is_container_magic_bytes_zip():
    # ZIP magic bytes: PK\x03\x04
    content = b'PK\x03\x04' + b'some data'
    assert Config.is_container("file.unknown", content=content) is True

def test_is_container_magic_bytes_gzip():
    # GZIP magic bytes: \x1f\x8b
    content = b'\x1f\x8b' + b'some data'
    assert Config.is_container("file.unknown", content=content) is True

def test_is_container_magic_bytes_tar():
    # TAR magic bytes: 'ustar' at offset 257
    content = b'A' * 257 + b'ustar' + b'B' * 10
    assert Config.is_container("file.unknown", content=content) is True

def test_is_container_magic_bytes_too_short():
    content = b'ustar' # Too short to check offset 257
    assert Config.is_container("file.unknown", content=content) is False

def test_is_container_by_extension():
    extensions = [
        '.zip', '.tar', '.tar.gz', '.ipynb', '.md', '.html', '.htm', '.xhtml',
        '.yml', '.yaml', '.diff', '.patch'
    ]
    for ext in extensions:
        assert Config.is_container(f"test{ext}") is True

def test_is_container_by_manifest_name():
    manifests = ['package.json', 'composer.json', 'deno.json', 'deno.jsonc', 'pyproject.toml']
    for name in manifests:
        assert Config.is_container(name) is True
        assert Config.is_container(f"sub/dir/{name}") is True

def test_is_container_manifest_variants():
    # Should match variants like my.package.json if consistency with unpack_content is desired
    variants = ['my.package.json', 'prod.composer.json', 'custom.deno.json']
    for name in variants:
        assert Config.is_container(name) is True

def test_is_container_notebook_by_content():
    content = b'{"cells": [], "metadata": {}}'
    assert Config.is_container("unknown_file", content=content) is True

    # Negative case: JSON but not a notebook
    content_not_notebook = b'{"key": "value"}'
    assert Config.is_container("unknown_file", content=content_not_notebook) is False

def test_is_container_by_devops_name():
    devops = ['Dockerfile', 'Makefile', 'my.dockerfile', 'project.makefile']
    for name in devops:
        assert Config.is_container(name) is True
        assert Config.is_container(name.lower()) is True
        assert Config.is_container(f"path/to/{name}") is True

def test_is_container_negative_cases():
    assert Config.is_container("script.py") is False
    assert Config.is_container("data.txt") is False
    assert Config.is_container("README") is False
    assert Config.is_container("file.unknown", content=b"just plain text") is False

def test_is_container_path_object():
    assert Config.is_container(Path("package.json")) is True
    assert Config.is_container(Path("test.zip")) is True
