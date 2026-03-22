import io
import json
import zipfile
import tarfile
import pytest
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
