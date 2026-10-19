import json
import pytest
from gptscan import generate_json, export_results_to_file

SAMPLE_RESULTS = [
    {
        "path": "src/vulnerable.py",
        "line": 42,
        "own_conf": "90%",
        "admin_desc": "Dangerous exec usage",
        "end-user_desc": "Code execution risk detected",
        "gpt_conf": "85%",
        "snippet": "exec(user_input)"
    },
    {
        "path": "lib/utils.js",
        "line": 10,
        "own_conf": "75%",
        "admin_desc": "Eval detected",
        "end-user_desc": "Risk detected",
        "gpt_conf": "80%",
        "snippet": "eval(code)"
    }
]


def test_generate_json_empty():
    """Verify generate_json with an empty list returns '[]'."""
    res = generate_json([])
    assert res.strip() == "[]"


def test_generate_json_structure():
    """Verify generate_json formats a list of result dicts into valid JSON string."""
    res = generate_json(SAMPLE_RESULTS)
    data = json.loads(res)
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["path"] == "src/vulnerable.py"
    assert data[1]["path"] == "lib/utils.js"


def test_export_results_to_file_json_array(tmp_path):
    """Verify export_results_to_file with output_format='json' produces a JSON array."""
    out_file = tmp_path / "results.json"
    export_results_to_file(str(out_file), SAMPLE_RESULTS, output_format="json")
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    data = json.loads(content)
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["line"] == 42
    assert data[1]["snippet"] == "eval(code)"
