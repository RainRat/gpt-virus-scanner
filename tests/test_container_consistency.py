import pytest
import json
from unittest.mock import MagicMock, patch
from pathlib import Path
import gptscan
from gptscan import scan_files, Config

def test_local_composer_json_unpacking(monkeypatch, tmp_path):
    """Test that a local composer.json is unpacked."""
    composer_data = {
        "scripts": {
            "test": "phpunit"
        }
    }
    composer_file = tmp_path / "composer.json"
    composer_file.write_text(json.dumps(composer_data))

    # Mock model to avoid TF
    mock_model = MagicMock()
    mock_model.predict.return_value = [[0.5]]
    monkeypatch.setattr(gptscan, "get_model", lambda: mock_model)
    monkeypatch.setattr(gptscan, "_tf_module", MagicMock())

    Config.set_extensions(Config.DEFAULT_EXTENSIONS)

    events = list(scan_files(
        scan_targets=[str(composer_file)],
        deep_scan=False,
        show_all=True,
        use_gpt=False
    ))

    results = [data for event, data in events if event == 'result']
    names = [r[0] for r in results]

    # If it's not unpacked, name will be just the path to composer.json
    # If it is unpacked, it should contain "[Script: test]"
    assert any("[Script: test]" in n for n in names), f"Expected unpacked script for local composer.json, but got names: {names}"

def test_local_deno_json_unpacking(monkeypatch, tmp_path):
    """Test that a local deno.json is unpacked."""
    deno_data = {
        "tasks": {
            "start": "deno run main.ts"
        }
    }
    deno_file = tmp_path / "deno.json"
    deno_file.write_text(json.dumps(deno_data))

    mock_model = MagicMock()
    mock_model.predict.return_value = [[0.5]]
    monkeypatch.setattr(gptscan, "get_model", lambda: mock_model)
    monkeypatch.setattr(gptscan, "_tf_module", MagicMock())

    Config.set_extensions(Config.DEFAULT_EXTENSIONS)

    events = list(scan_files(
        scan_targets=[str(deno_file)],
        deep_scan=False,
        show_all=True,
        use_gpt=False
    ))

    results = [data for event, data in events if event == 'result']
    names = [r[0] for r in results]

    assert any("[Task: start]" in n for n in names), f"Expected unpacked task for local deno.json, but got names: {names}"
