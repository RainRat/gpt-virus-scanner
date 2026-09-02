import io
import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

import gptscan


def test_export_results_to_file_tsv(tmp_path):
    """Verify that export_results_to_file correctly formats findings as TSV."""
    out_file = tmp_path / "results.tsv"
    results = [
        {
            "path": "test_script.py",
            "own_conf": "85%",
            "admin_desc": "Suspicious execution",
            "end-user_desc": "Potential malware",
            "gpt_conf": "90%",
            "snippet": "import os; os.system('ls')",
            "line": 10
        }
    ]

    gptscan.export_results_to_file(str(out_file), results, output_format='tsv')

    assert out_file.exists()
    content = out_file.read_text(encoding='utf-8')
    lines = [line for line in content.splitlines() if line.strip()]

    # Header check
    assert lines[0] == "path\town_conf\tadmin_desc\tend-user_desc\tgpt_conf\tsnippet\tline"
    # Row check
    assert lines[1] == "test_script.py\t85%\tSuspicious execution\tPotential malware\t90%\timport os; os.system('ls')\t10"


def test_parse_report_content_tsv():
    """Verify parse_report_content parses TSV content with filename_hint or tab detection."""
    tsv_content = (
        "path\town_conf\tadmin_desc\tend-user_desc\tgpt_conf\tsnippet\tline\n"
        "script.py\t75%\tAdmin note\tUser note\t80%\tprint('hello')\t5\n"
    )

    # Test with hint
    results_hint = gptscan.parse_report_content(tsv_content, filename_hint="report.tsv")
    assert len(results_hint) == 1
    item = results_hint[0]
    assert item["path"] == "script.py"
    assert item["own_conf"] == "75%"
    assert item["admin_desc"] == "Admin note"
    assert item["end-user_desc"] == "User note"
    assert item["gpt_conf"] == "80%"
    assert item["snippet"] == "print('hello')"
    assert item["line"] == "5"

    # Test auto-detection without hint
    results_auto = gptscan.parse_report_content(tsv_content)
    assert len(results_auto) == 1
    assert results_auto[0]["path"] == "script.py"


def test_load_report_file_tsv(tmp_path):
    """Verify load_report_file loads a TSV file or scans directory containing TSV files."""
    tsv_file = tmp_path / "sample.tsv"
    tsv_content = (
        "path\town_conf\tadmin_desc\tend-user_desc\tgpt_conf\tsnippet\tline\n"
        "sample.py\t90%\tAdmin test\tUser test\t95%\tos.remove('foo')\t12\n"
    )
    tsv_file.write_text(tsv_content, encoding='utf-8')

    # Load single file
    file_results = gptscan.load_report_file(str(tsv_file))
    assert len(file_results) == 1
    assert file_results[0]["path"] == "sample.py"

    # Load directory containing TSV
    dir_results = gptscan.load_report_file(str(tmp_path))
    assert len(dir_results) == 1
    assert dir_results[0]["path"] == "sample.py"


def test_run_cli_tsv_output(tmp_path, monkeypatch):
    """Verify run_cli outputs results in TSV format when output_format='tsv'."""
    out_file = tmp_path / "cli_out.tsv"
    test_file = tmp_path / "malicious.py"
    test_file.write_text("import os; os.system('rm -rf /')\n", encoding='utf-8')

    # Mock prediction model to generate threat level
    monkeypatch.setattr(gptscan, "get_model", lambda: MagicMock())
    monkeypatch.setattr(gptscan, "_tf_module", MagicMock())

    def mock_scan_files(*args, **kwargs):
        yield ('result', (str(test_file), '80%', 'Command execution', 'Dangerous command', '85%', "os.system('rm -rf /')", 1))
        yield ('summary', (1, 100, 0.1))

    monkeypatch.setattr(gptscan, "scan_files", mock_scan_files)

    ret = gptscan.run_cli(
        targets=[str(test_file)],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        output_format='tsv',
        output_file=str(out_file),
        quiet=True
    )

    assert ret == 1
    assert out_file.exists()
    content = out_file.read_text(encoding='utf-8')
    assert "\t" in content
    lines = content.strip().splitlines()
    assert lines[0].startswith("path\town_conf\t")
    assert str(test_file) in lines[1]


def test_export_results_gui_tsv(monkeypatch, tmp_path):
    """Verify export_results GUI command works with .tsv extension."""
    out_file = tmp_path / "export.tsv"
    sample_results = [
        {"path": "file.py", "own_conf": "50%", "admin_desc": "a", "end-user_desc": "u", "gpt_conf": "50%", "snippet": "code", "line": "1"}
    ]

    monkeypatch.setattr(gptscan.tkinter.filedialog, "asksaveasfilename", lambda **k: str(out_file))
    monkeypatch.setattr(gptscan, "_get_tree_results_as_dicts", lambda item_ids: sample_results)
    monkeypatch.setattr(gptscan, "tree", MagicMock())
    monkeypatch.setattr(gptscan, "messagebox", MagicMock())

    gptscan.export_results()

    assert out_file.exists()
    content = out_file.read_text(encoding='utf-8')
    assert "file.py\t50%\ta\tu\t50%\tcode\t1" in content
