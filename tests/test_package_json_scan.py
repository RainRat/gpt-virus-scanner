import json
import pytest
from gptscan import unpack_content

def test_unpack_package_json_scripts():
    """Verify that scripts are correctly extracted from package.json files."""
    pkg_data = {
        "name": "test-app",
        "version": "1.0.0",
        "scripts": {
            "start": "node index.js",
            "test": "jest",
            "build": "webpack --mode production",
            "empty": "  ",
            "non-string": 123
        }
    }
    content_bytes = json.dumps(pkg_data).encode('utf-8')

    # Test with package.json filename
    snippets = list(unpack_content("package.json", content_bytes))

    assert len(snippets) == 3

    # Check each extracted script
    script_names = [s[0] for s in snippets]
    assert "package.json [Script: start]" in script_names
    assert "package.json [Script: test]" in script_names
    assert "package.json [Script: build]" in script_names

    # Verify content
    scripts_dict = {name: cmd for name, cmd in snippets}
    assert scripts_dict["package.json [Script: start]"] == b"node index.js"
    assert scripts_dict["package.json [Script: test]"] == b"jest"
    assert scripts_dict["package.json [Script: build]"] == b"webpack --mode production"

def test_unpack_package_json_no_scripts():
    """Verify that package.json files with no scripts yield no snippets."""
    pkg_data = {"name": "no-scripts"}
    content_bytes = json.dumps(pkg_data).encode('utf-8')

    snippets = list(unpack_content("package.json", content_bytes))
    assert len(snippets) == 0

def test_unpack_package_json_invalid_scripts():
    """Verify that package.json with non-dictionary scripts yields no snippets."""
    pkg_data = {"scripts": "not-a-dict"}
    content_bytes = json.dumps(pkg_data).encode('utf-8')

    snippets = list(unpack_content("package.json", content_bytes))
    assert len(snippets) == 0

def test_unpack_package_json_empty():
    """Verify that an empty or malformed package.json is handled gracefully."""
    snippets = list(unpack_content("package.json", b""))
    assert len(snippets) == 0

    snippets = list(unpack_content("package.json", b"{invalid json}"))
    assert len(snippets) == 0

def test_unpack_nested_package_json():
    """Verify that package.json scripts are extracted even when nested in an archive name."""
    pkg_data = {"scripts": {"dev": "nodemon server.js"}}
    content_bytes = json.dumps(pkg_data).encode('utf-8')

    snippets = list(unpack_content("my-project[package.json]", content_bytes))
    assert len(snippets) == 1
    assert snippets[0][0] == "my-project[package.json] [Script: dev]"
    assert snippets[0][1] == b"nodemon server.js"
