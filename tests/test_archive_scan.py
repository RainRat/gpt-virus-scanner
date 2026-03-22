import zipfile
import tarfile
import io
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import gptscan
from gptscan import scan_files, Config

def test_zip_scanning(mock_tf_env, tmp_path):
    """Test that scripts inside a ZIP file are detected and scanned."""
    mock_tf_env.predict.return_value = [[0.9]]
    zip_path = tmp_path / "test.zip"
    script_content = b"print('malicious zip')"

    with zipfile.ZipFile(zip_path, 'w') as z:
        z.writestr("malicious.py", script_content)
        z.writestr("safe.txt", b"this is a safe text file")

    events = list(scan_files(
        scan_targets=[str(zip_path)],
        deep_scan=False,
        show_all=True,
        use_gpt=False
    ))

    results = [data for event, data in events if event == 'result']
    # malicious.py should be found, safe.txt should NOT (it's not a script)
    assert len(results) == 1
    path, own_conf, admin, user, gpt, snippet, line = results[0]
    assert "test.zip[malicious.py]" in path
    assert own_conf == "90%"
    assert "print('malicious zip')" in snippet

def test_tar_scanning(mock_tf_env, tmp_path):
    """Test that scripts inside a TAR file are detected and scanned."""
    mock_tf_env.predict.return_value = [[0.9]]
    tar_path = tmp_path / "test.tar"
    script_content = b"#!/bin/bash\necho 'malicious tar'"

    with tarfile.open(tar_path, 'w') as t:
        info = tarfile.TarInfo("malicious.sh")
        info.size = len(script_content)
        t.addfile(info, io.BytesIO(script_content))

        info2 = tarfile.TarInfo("readme.md")
        info2.size = len(b"safe")
        t.addfile(info2, io.BytesIO(b"safe"))

    events = list(scan_files(
        scan_targets=[str(tar_path)],
        deep_scan=False,
        show_all=True,
        use_gpt=False
    ))

    results = [data for event, data in events if event == 'result']
    assert len(results) == 1
    path, own_conf, admin, user, gpt, snippet, line = results[0]
    assert "test.tar[malicious.sh]" in path
    assert own_conf == "90%"
    assert "echo 'malicious tar'" in snippet

def test_nested_unsupported_archive(mock_tf_env, tmp_path):
    """Test that non-archive files are still handled correctly when mixed with archives."""
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as z:
        z.writestr("script.py", b"print('zip')")

    regular_script = tmp_path / "regular.py"
    regular_script.write_bytes(b"print('regular')")

    events = list(scan_files(
        scan_targets=[str(tmp_path)],
        deep_scan=False,
        show_all=True,
        use_gpt=False
    ))

    results = {data[0]: data for event, data in events if event == 'result'}
    assert len(results) == 2
    assert any("test.zip[script.py]" in k for k in results)
    assert any("regular.py" in k for k in results)
