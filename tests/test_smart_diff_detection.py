import pytest
from gptscan import unpack_content, Config

def test_clipboard_diff_unpacked_by_content():
    """Verify that diffs without extensions (like from clipboard) are correctly unpacked."""
    content = b"--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,2 @@\n+new line"
    results = list(unpack_content("[Clipboard]", content))

    # It should be unpacked into a hunk
    assert len(results) == 1
    assert results[0][0] == "[Clipboard] [file.py @ line 1]"
    assert b"+new line" in results[0][1]

def test_extensionless_git_diff_unpacked():
    """Verify that 'diff --git' style diffs are recognized by content."""
    content = b"diff --git a/test.js b/test.js\nindex 123..456 100644\n--- a/test.js\n+++ b/test.js\n@@ -1,1 +1,1 @@\n-old\n+new"
    results = list(unpack_content("raw_diff", content))

    assert len(results) == 1
    assert results[0][0] == "raw_diff [test.js @ line 1]"
    assert b"+new" in results[0][1]

def test_docker_compose_is_container():
    """Verify that docker-compose files are recognized as containers for unpacking."""
    assert Config.is_container("docker-compose.yml") is True
    assert Config.is_container("docker-compose.yaml") is True
    assert Config.is_container("project/docker-compose.yml") is True
    assert Config.is_container("my.docker-compose.yml") is True
