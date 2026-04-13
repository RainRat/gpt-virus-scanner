from pathlib import Path
from gptscan import Config

def test_is_container_magic_bytes_zip():
    content = b'PK\x03\x04some content'
    assert Config.is_container("somefile", content=content) is True

def test_is_container_magic_bytes_gzip():
    content = b'\x1f\x8bsome content'
    assert Config.is_container("somefile", content=content) is True

def test_is_container_magic_bytes_tar():
    # Create a dummy TAR header with 'ustar' at 257
    content = b'\0' * 257 + b'ustar' + b'\0' * 10
    assert Config.is_container("somefile", content=content) is True

def test_is_container_extensions():
    assert Config.is_container("test.zip") is True
    assert Config.is_container("test.tar") is True
    assert Config.is_container("test.tar.gz") is True
    assert Config.is_container("test.ipynb") is True
    assert Config.is_container("test.md") is True
    assert Config.is_container("test.html") is True
    assert Config.is_container("package.json") is True
    assert Config.is_container("composer.json") is True
    assert Config.is_container("deno.json") is True
    assert Config.is_container("deno.jsonc") is True
    assert Config.is_container("test.diff") is True
    assert Config.is_container("test.patch") is True

def test_is_container_basenames():
    assert Config.is_container("Dockerfile") is True
    assert Config.is_container("dockerfile") is True
    assert Config.is_container("Makefile") is True
    assert Config.is_container("makefile") is True
    assert Config.is_container("prod.dockerfile") is True
    assert Config.is_container("module.makefile") is True

def test_is_container_path_variants():
    assert Config.is_container(Path("TEST.ZIP")) is True
    assert Config.is_container("sub/dir/PACKAGE.JSON") is True

def test_is_container_negatives():
    assert Config.is_container("test.py") is False
    assert Config.is_container("test.js") is False
    assert Config.is_container("test.txt") is False
    assert Config.is_container("README") is False
    # Content that doesn't match magic
    assert Config.is_container("test.txt", content=b"hello world") is False

def test_is_supported_file_delegation_to_container():
    # ZIP magic bytes should make it supported even if extension is .txt
    content = b'PK\x03\x04some content'
    assert Config.is_supported_file("test.txt", content=content) is True

    # TAR magic bytes
    content = b'\0' * 257 + b'ustar' + b'\0' * 10
    assert Config.is_supported_file("test.txt", content=content) is True
