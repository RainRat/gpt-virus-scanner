import io
import json
import zipfile
import tarfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import gptscan
from gptscan import unpack_content, scan_files, Config

def test_unpack_zip_in_memory():
    """Test that unpack_content correctly expands a ZIP buffer."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as z:
        z.writestr("test.py", "print('hello')")
        z.writestr("other.txt", "not a script")

    zip_content = zip_buffer.getvalue()
    results = list(unpack_content("test.zip", zip_content))

    assert len(results) == 1
    assert results[0][0] == "test.zip[test.py]"
    assert results[0][1] == b"print('hello')"

def test_unpack_tar_in_memory():
    """Test that unpack_content correctly expands a TAR buffer."""
    tar_buffer = io.BytesIO()
    script_content = b"print('tar')"

    with tarfile.open(fileobj=tar_buffer, mode='w') as t:
        info = tarfile.TarInfo("test.py")
        info.size = len(script_content)
        t.addfile(info, io.BytesIO(script_content))

        info2 = tarfile.TarInfo("other.txt")
        info2.size = 5
        t.addfile(info2, io.BytesIO(b"other"))

    tar_content = tar_buffer.getvalue()
    results = list(unpack_content("test.tar", tar_content))

    assert len(results) == 1
    assert results[0][0] == "test.tar[test.py]"
    assert results[0][1] == b"print('tar')"

def test_unpack_tgz_in_memory():
    """Test that unpack_content correctly expands a TAR.GZ buffer via magic bytes."""
    tgz_buffer = io.BytesIO()
    script_content = b"print('tgz')"

    with tarfile.open(fileobj=tgz_buffer, mode='w:gz') as t:
        info = tarfile.TarInfo("test.py")
        info.size = len(script_content)
        t.addfile(info, io.BytesIO(script_content))

    tgz_content = tgz_buffer.getvalue()
    # TGZ doesn't have the TAR magic bytes at offset 257 in the compressed stream
    # but starts with GZIP magic bytes b'\x1f\x8b'
    results = list(unpack_content("test.tar.gz", tgz_content))

    assert len(results) == 1
    assert results[0][0] == "test.tar.gz[test.py]"
    assert results[0][1] == b"print('tgz')"

def test_unpack_markdown_blocks():
    """Test that unpack_content correctly extracts code blocks from Markdown."""
    md_content = b"""
# My Markdown
```python
print('block 1')
```
Some text.
```bash
echo 'block 2'
```
"""
    results = list(unpack_content("readme.md", md_content))

    assert len(results) == 2
    assert results[0][0] == "readme.md [Block 1]"
    assert b"print('block 1')" in results[0][1]
    assert results[1][0] == "readme.md [Block 2]"
    assert b"echo 'block 2'" in results[1][1]

def test_is_supported_file_with_content():
    """Test that is_supported_file correctly uses the provided content for shebang detection."""
    content = b"#!/usr/bin/python\nprint('hi')"
    # Even with a non-script extension, it should return True because of the content shebang
    assert Config.is_supported_file("test.txt", content=content) is True

    # Without content, it would be False
    assert Config.is_supported_file("test.txt") is False

def test_unpack_ipynb_in_memory():
    """Test that unpack_content correctly parses a Notebook buffer."""
    nb_data = {
        "cells": [
            {"cell_type": "code", "source": ["print('cell 1')"]},
            {"cell_type": "markdown", "source": ["# Title"]},
            {"cell_type": "code", "source": ["import os\n", "os.system('ls')"]}
        ]
    }
    nb_content = json.dumps(nb_data).encode('utf-8')
    results = list(unpack_content("test.ipynb", nb_content))

    assert len(results) == 2
    assert results[0][0] == "test.ipynb [Cell 1]"
    assert b"cell 1" in results[0][1]
    assert results[1][0] == "test.ipynb [Cell 2]"
    assert b"os.system" in results[1][1]

def test_unpack_nested_archives():
    """Test that unpack_content handles nested archives (ZIP inside ZIP)."""
    inner_zip_buffer = io.BytesIO()
    with zipfile.ZipFile(inner_zip_buffer, 'w') as z:
        z.writestr("inner.py", "print('inner')")

    outer_zip_buffer = io.BytesIO()
    with zipfile.ZipFile(outer_zip_buffer, 'w') as z:
        z.writestr("nested.zip", inner_zip_buffer.getvalue())

    results = list(unpack_content("outer.zip", outer_zip_buffer.getvalue()))

    assert len(results) == 1
    assert results[0][0] == "outer.zip[nested.zip][inner.py]"
    assert results[0][1] == b"print('inner')"

def test_scan_files_url_zip_expansion(mock_tf_env, monkeypatch):
    """Test that a ZIP fetched via URL is expanded and scanned."""
    monkeypatch.setattr(gptscan, "collect_files", lambda targets: [])
    url = "https://example.com/scripts.zip"

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as z:
        z.writestr("malicious.py", "eval(input())")

    mock_content = zip_buffer.getvalue()

    mock_response = MagicMock()
    mock_response.getheader.return_value = str(len(mock_content))
    mock_response.read.return_value = mock_content
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        events = list(scan_files(
            scan_targets=[url],
            deep_scan=False,
            show_all=True,
            use_gpt=False
        ))

    results = [data for event, data in events if event == 'result']
    # Should find malicious.py inside the ZIP
    assert any("scripts.zip[malicious.py]" in r[0] for r in results)

def test_scan_files_clipboard_ipynb_expansion(mock_tf_env, monkeypatch):
    """Test that a Notebook in extra_snippets (clipboard) is expanded."""
    monkeypatch.setattr(gptscan, "collect_files", lambda targets: [])
    nb_data = {
        "cells": [{"cell_type": "code", "source": ["print('clipboard cell')"]}]
    }
    nb_content = json.dumps(nb_data).encode('utf-8')

    events = list(scan_files(
        scan_targets=[],
        deep_scan=False,
        show_all=True,
        use_gpt=False,
        extra_snippets=[("[Clipboard]", nb_content)]
    ))

    results = [data for event, data in events if event == 'result']
    assert len(results) == 1
    assert "[Clipboard] [Cell 1]" in results[0][0]
    assert b"clipboard cell" in results[0][5].encode('utf-8')
