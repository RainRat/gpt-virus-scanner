import os
import pytest
from gptscan import get_environment_variable_snippets

def test_get_environment_variable_snippets(monkeypatch):
    # Setup mock environment variables
    monkeypatch.setenv("TEST_VAR_1", "dangerous_command")
    monkeypatch.setenv("TEST_VAR_2", "  ") # Should be ignored (whitespace only)
    monkeypatch.setenv("EMPTY_VAR", "")    # Should be ignored

    snippets = get_environment_variable_snippets()

    # Check if TEST_VAR_1 is present
    found_var1 = False
    for name, content in snippets:
        if name == "[EnvVar] TEST_VAR_1":
            assert content == b"dangerous_command"
            found_var1 = True
        elif name == "[EnvVar] TEST_VAR_2":
            pytest.fail("TEST_VAR_2 should have been ignored (whitespace only)")
        elif name == "[EnvVar] EMPTY_VAR":
            pytest.fail("EMPTY_VAR should have been ignored")

    assert found_var1, "TEST_VAR_1 not found in snippets"

def test_cli_env_vars_flag(monkeypatch):
    import subprocess
    import sys

    # We'll use subprocess to run the CLI to ensure full integration
    # Set a unique env var to look for
    env = os.environ.copy()
    env["GPT_SCAN_TEST_ENV"] = "suspicious_payload"

    # Run gptscan.py with --env-vars --cli --dry-run
    # --dry-run ensures we don't actually need the model file
    # We use a non-existent target to only scan the environment variables
    cmd = [sys.executable, "gptscan.py", "non_existent_target_999", "--env-vars", "--cli", "--dry-run"]

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    # Check both stdout and stderr because it might depend on terminal vs redirection
    combined_output = result.stdout + result.stderr

    assert "GPT_SCAN_TEST_ENV" in combined_output
    # Count should be 1 if only environment variables are scanned
    # But get_environment_variable_snippets returns ALL env vars, so we check for many files
    assert "Scan complete" in combined_output
    assert "0 suspicious files found" in combined_output
