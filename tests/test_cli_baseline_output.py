import json
import pytest
from unittest.mock import MagicMock
import gptscan


@pytest.fixture(autouse=True)
def mock_keras_model(monkeypatch):
    """Mock the Keras model and TensorFlow module to avoid Keras 3 / TensorFlow deserialization issues on Python 3.12."""
    mock_model = MagicMock()
    # Predict returns a high threat confidence array (e.g. [[0.9]]) for any 1024-byte window
    mock_model.predict.return_value = [[0.9]]
    mock_tf = MagicMock()
    monkeypatch.setattr(gptscan, "get_model", lambda: mock_model)
    monkeypatch.setattr(gptscan, "_tf_module", mock_tf)


def test_run_cli_baseline_output(tmp_path):
    """Test that run_cli exports bypassed baseline findings to baseline_output_file."""
    baseline_file = tmp_path / "baseline.json"
    baseline_data = [
        {
            "path": "suspicious.py",
            "own_conf": "90%",
            "admin_desc": "Known threat",
            "end-user_desc": "Known threat",
            "gpt_conf": "",
            "snippet": "import os\nos.system('rm -rf /')",
            "line": "1"
        }
    ]
    baseline_file.write_text(json.dumps(baseline_data), encoding="utf-8")

    baseline_out = tmp_path / "bypassed.json"
    main_out = tmp_path / "main.json"

    snippets = [("suspicious.py", b"import os\nos.system('rm -rf /')")]

    threats = gptscan.run_cli(
        targets=[],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        output_file=str(main_out),
        extra_snippets=snippets,
        baseline_file=str(baseline_file),
        baseline_output_file=str(baseline_out),
        quiet=True
    )

    # 1. Main threat count should be 0 because it was bypassed by baseline
    assert threats == 0

    # 2. Main output file should be empty (or contain no findings)
    assert main_out.exists()
    assert main_out.read_text(encoding="utf-8").strip() == ""

    # 3. Bypassed baseline output file should contain the bypassed finding
    assert baseline_out.exists()
    bypassed_text = baseline_out.read_text(encoding="utf-8").strip()
    assert bypassed_text != ""
    record = json.loads(bypassed_text)
    assert record["path"] == "suspicious.py"
    assert "os.system" in record["snippet"]


def test_run_cli_baseline_output_csv_format(tmp_path):
    """Test that baseline_output_file works with CSV output format."""
    baseline_file = tmp_path / "baseline.csv"
    baseline_content = 'path,line,own_conf,gpt_conf,admin_desc,end-user_desc,snippet\nsuspicious.py,1,90%,,,,"import os\nos.system(\'rm -rf /\')"\n'
    baseline_file.write_text(baseline_content, encoding="utf-8")

    baseline_out = tmp_path / "bypassed.csv"

    snippets = [("suspicious.py", b"import os\nos.system('rm -rf /')")]

    threats = gptscan.run_cli(
        targets=[],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_format="csv",
        extra_snippets=snippets,
        baseline_file=str(baseline_file),
        baseline_output_file=str(baseline_out),
        quiet=True
    )

    assert threats == 0
    assert baseline_out.exists()
    bypassed_text = baseline_out.read_text(encoding="utf-8")
    assert "suspicious.py" in bypassed_text
    assert "path,own_conf,admin_desc" in bypassed_text or "path,line,own_conf" in bypassed_text


def test_run_cli_baseline_output_no_matches(tmp_path):
    """Test baseline_output_file when no findings match the baseline."""
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text("[]", encoding="utf-8")

    baseline_out = tmp_path / "bypassed.json"

    snippets = [("suspicious.py", b"import os\nos.system('rm -rf /')")]

    threats = gptscan.run_cli(
        targets=[],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        extra_snippets=snippets,
        baseline_file=str(baseline_file),
        baseline_output_file=str(baseline_out),
        quiet=True
    )

    assert threats == 1
    assert baseline_out.exists()
    assert baseline_out.read_text(encoding="utf-8").strip() == ""
