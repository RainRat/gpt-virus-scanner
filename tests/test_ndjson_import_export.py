import json
from unittest.mock import MagicMock
import pytest
import gptscan


def test_export_results_to_file_ndjson(tmp_path):
    """Verify that export_results_to_file with output_format='ndjson' writes line-by-line JSON records."""
    file_path = tmp_path / "results.ndjson"
    sample_results = [
        {
            "path": "script1.py",
            "line": 10,
            "own_conf": "80%",
            "admin_desc": "Suspicious eval",
            "end-user_desc": "Malicious code detected",
            "gpt_conf": "85%",
            "snippet": "eval(code)"
        },
        {
            "path": "script2.js",
            "line": 5,
            "own_conf": "90%",
            "admin_desc": "Obfuscated payload",
            "end-user_desc": "Risk found",
            "gpt_conf": "95%",
            "snippet": "atob(data)"
        }
    ]

    gptscan.export_results_to_file(str(file_path), sample_results, output_format="ndjson")

    content = file_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    assert len(lines) == 2

    record1 = json.loads(lines[0])
    assert record1["path"] == "script1.py"
    assert record1["gpt_conf"] == "85%"

    record2 = json.loads(lines[1])
    assert record2["path"] == "script2.js"
    assert record2["snippet"] == "atob(data)"


def test_export_results_to_file_jsonl(tmp_path):
    """Verify that export_results_to_file accepts 'jsonl' format alias."""
    file_path = tmp_path / "results.jsonl"
    sample_results = [
        {
            "path": "test.py",
            "line": 1,
            "own_conf": "50%",
            "admin_desc": "Note",
            "end-user_desc": "Note",
            "gpt_conf": "",
            "snippet": "print(1)"
        }
    ]

    gptscan.export_results_to_file(str(file_path), sample_results, output_format="jsonl")

    content = file_path.read_text(encoding="utf-8").strip()
    record = json.loads(content)
    assert record["path"] == "test.py"


def test_export_results_gui_ndjson(monkeypatch, tmp_path):
    """Verify GUI export_results function handles .ndjson files."""
    file_path = tmp_path / "export.ndjson"
    monkeypatch.setattr(gptscan.filedialog, 'asksaveasfilename', lambda **k: str(file_path))

    mock_tree = MagicMock()
    mock_tree.get_children.return_value = ["item1"]
    monkeypatch.setattr(gptscan, "tree", mock_tree)

    def mock_get_dicts(children):
        return [{
            "path": "gui_test.py",
            "line": 12,
            "own_conf": "75%",
            "admin_desc": "Admin note",
            "end-user_desc": "User note",
            "gpt_conf": "80%",
            "snippet": "import os"
        }]

    monkeypatch.setattr(gptscan, "_get_tree_results_as_dicts", mock_get_dicts)
    mock_msg = MagicMock()
    monkeypatch.setattr(gptscan.messagebox, 'showinfo', mock_msg)

    gptscan.export_results()

    assert file_path.exists()
    lines = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["path"] == "gui_test.py"
    assert mock_msg.called


def test_ndjson_cli_flag_output(capsys, monkeypatch):
    """Verify CLI --ndjson flag outputs NDJSON formatted findings."""
    def mock_scan_files(*args, **kwargs):
        yield ('result', ('suspicious.py', '80%', 'Admin note', 'User note', '85%', 'os.system()', 15))
        yield ('summary', (1, 100, 0.1))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    threats = gptscan.run_cli(
        targets=["."],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_format="ndjson",
        quiet=True
    )

    assert threats == 1
    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.splitlines() if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["path"] == "suspicious.py"
    assert record["line"] == 15


def test_ndjson_file_extension_inference(tmp_path, monkeypatch):
    """Verify that saving output to a .ndjson or .jsonl file automatically sets output_format='ndjson'."""
    out_file = tmp_path / "output.ndjson"

    def mock_scan_files(*args, **kwargs):
        yield ('result', ('test_infer.py', '60%', 'Note A', 'Note B', '70%', 'exec()', 42))
        yield ('summary', (1, 100, 0.1))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    gptscan.run_cli(
        targets=["."],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_file=str(out_file),
        output_format="ndjson",
        quiet=True
    )

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8").strip()
    record = json.loads(content)
    assert record["path"] == "test_infer.py"


def test_ndjson_roundtrip_import(tmp_path):
    """Verify that an exported NDJSON file can be imported back into result dicts via load_report_file."""
    file_path = tmp_path / "roundtrip.ndjson"
    original_data = [
        {
            "path": "app.py",
            "line": 10,
            "own_conf": "90%",
            "admin_desc": "Critical finding",
            "end-user_desc": "High threat detected",
            "gpt_conf": "95%",
            "snippet": "subprocess.call(cmd)"
        }
    ]

    gptscan.export_results_to_file(str(file_path), original_data, output_format="ndjson")

    imported = gptscan.load_report_file(str(file_path))
    assert len(imported) == 1
    assert imported[0]["path"] == "app.py"
    assert str(imported[0]["line"]) == "10"
    assert imported[0]["own_conf"] == "90%"
