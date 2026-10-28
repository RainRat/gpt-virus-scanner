import csv
import io
import pytest
from unittest.mock import MagicMock
import gptscan


def test_generate_csv_default_fields():
    results = [
        {
            "path": "script.py",
            "line": 10,
            "own_conf": "85%",
            "gpt_conf": "90%",
            "admin_desc": "Admin note",
            "end-user_desc": "User note",
            "snippet": "os.system('rm -rf /')",
            "extra_field": "ignored"
        }
    ]
    output = gptscan.generate_csv(results)
    lines = output.strip().splitlines()

    assert len(lines) == 2
    assert lines[0] == "path,line,own_conf,gpt_conf,admin_desc,end-user_desc,snippet"
    assert "script.py,10,85%,90%,Admin note,User note,os.system('rm -rf /')" in lines[1]


def test_generate_csv_tsv_delimiter():
    results = [
        {
            "path": "test.js",
            "line": 1,
            "own_conf": "50%",
            "gpt_conf": "0%",
            "admin_desc": "Note 1",
            "end-user_desc": "Note 2",
            "snippet": "console.log('hi')"
        }
    ]
    output = gptscan.generate_csv(results, delimiter='\t')
    lines = output.strip().splitlines()

    assert len(lines) == 2
    assert lines[0] == "path\tline\town_conf\tgpt_conf\tadmin_desc\tend-user_desc\tsnippet"
    assert lines[1] == "test.js\t1\t50%\t0%\tNote 1\tNote 2\tconsole.log('hi')"


def test_generate_csv_custom_fields():
    results = [
        {"path": "app.py", "line": 42, "own_conf": "95%"}
    ]
    custom_fields = ["path", "own_conf"]
    output = gptscan.generate_csv(results, fields=custom_fields)
    lines = output.strip().splitlines()

    assert len(lines) == 2
    assert lines[0] == "path,own_conf"
    assert lines[1] == "app.py,95%"


def test_generate_csv_empty_results():
    output = gptscan.generate_csv([])
    lines = output.strip().splitlines()

    assert len(lines) == 1
    assert lines[0] == "path,line,own_conf,gpt_conf,admin_desc,end-user_desc,snippet"


def test_copy_as_csv(monkeypatch):
    mock_tree = MagicMock()
    mock_tree.selection.return_value = ['item1']
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    results = [
        {
            "path": "demo.py",
            "line": 5,
            "own_conf": "80%",
            "gpt_conf": "85%",
            "admin_desc": "Suspicious call",
            "end-user_desc": "Risk detected",
            "snippet": "eval(code)"
        }
    ]
    monkeypatch.setattr(gptscan, '_get_tree_results_as_dicts', lambda sel: results)
    status_mock = MagicMock()
    monkeypatch.setattr(gptscan, 'update_status', status_mock)

    gptscan.copy_as_csv()

    mock_tree.clipboard_clear.assert_called_once()
    mock_tree.clipboard_append.assert_called_once()
    appended_text = mock_tree.clipboard_append.call_args[0][0]
    assert "demo.py,5,80%,85%,Suspicious call,Risk detected,eval(code)" in appended_text
    status_mock.assert_called_once_with("Copied 1 item(s) as CSV.")


def test_copy_as_tsv(monkeypatch):
    mock_tree = MagicMock()
    mock_tree.selection.return_value = ['item1', 'item2']
    monkeypatch.setattr(gptscan, 'tree', mock_tree)

    results = [
        {"path": "a.py", "line": 1, "own_conf": "10%", "gpt_conf": "0%", "admin_desc": "A", "end-user_desc": "A", "snippet": "a"},
        {"path": "b.py", "line": 2, "own_conf": "20%", "gpt_conf": "0%", "admin_desc": "B", "end-user_desc": "B", "snippet": "b"}
    ]
    monkeypatch.setattr(gptscan, '_get_tree_results_as_dicts', lambda sel: results)
    status_mock = MagicMock()
    monkeypatch.setattr(gptscan, 'update_status', status_mock)

    gptscan.copy_as_tsv()

    mock_tree.clipboard_clear.assert_called_once()
    mock_tree.clipboard_append.assert_called_once()
    appended_text = mock_tree.clipboard_append.call_args[0][0]
    assert "a.py\t1\t10%\t0%\tA\tA\ta" in appended_text
    assert "b.py\t2\t20%\t0%\tB\tB\tb" in appended_text
    status_mock.assert_called_once_with("Copied 2 item(s) as TSV.")
