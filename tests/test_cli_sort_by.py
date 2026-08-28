import io
import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

import gptscan
from gptscan import run_cli, main, Config


def test_run_cli_sort_by_threat(monkeypatch, tmp_path, capsys, mock_tf_env):
    f1 = tmp_path / "a.py"
    f1.write_text("import os; os.system('rm -rf /')\n")
    f2 = tmp_path / "b.py"
    f2.write_text("print('hello world')\n")

    # Mock model predict to give high threat score for f1 and low threat score for f2
    mock_model = MagicMock()
    def mock_predict(inputs, *args, **kwargs):
        # f1 contains rm -rf
        if isinstance(inputs, list):
            inp_bytes = bytes(inputs)
        else:
            inp_bytes = bytes(inputs[0]) if hasattr(inputs, '__getitem__') else b""
        return [[0.95]] if b"rm -rf" in inp_bytes else [[0.10]]

    mock_model.predict = mock_predict
    monkeypatch.setattr(gptscan, "get_model", lambda: mock_model)

    run_cli(
        targets=[str(f1), str(f2)],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        quiet=True,
        sort_by="threat"
    )

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
    assert len(lines) >= 2
    parsed = [json.loads(line) for line in lines]

    conf0 = int(parsed[0]['own_conf'].rstrip('%'))
    conf1 = int(parsed[1]['own_conf'].rstrip('%'))
    assert conf0 >= conf1


def test_run_cli_sort_by_path(tmp_path, capsys, mock_tf_env):
    f_z = tmp_path / "z_file.py"
    f_z.write_text("import os; os.system('evil')\n")
    f_a = tmp_path / "a_file.py"
    f_a.write_text("import os; os.system('evil')\n")

    run_cli(
        targets=[str(f_z), str(f_a)],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        quiet=True,
        sort_by="path"
    )

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]['path'].lower() < parsed[1]['path'].lower()


def test_run_cli_sort_by_line(tmp_path, capsys):
    import_report = tmp_path / "report.json"
    data = [
        {"path": "file.py", "line": "150", "own_conf": "80%", "admin_desc": "test", "end-user_desc": "test", "gpt_conf": "80%", "snippet": "code"},
        {"path": "file.py", "line": "10", "own_conf": "80%", "admin_desc": "test", "end-user_desc": "test", "gpt_conf": "80%", "snippet": "code"},
        {"path": "file.py", "line": "42", "own_conf": "80%", "admin_desc": "test", "end-user_desc": "test", "gpt_conf": "80%", "snippet": "code"},
    ]
    import_report.write_text(json.dumps(data))

    run_cli(
        targets=[],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        output_format="json",
        quiet=True,
        import_file=str(import_report),
        sort_by="line"
    )

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    lines_parsed = [int(p['line']) for p in parsed]
    assert lines_parsed == [10, 42, 150]


def test_run_cli_sort_by_csv_format(tmp_path, capsys):
    import_report = tmp_path / "report.json"
    data = [
        {"path": "z_file.py", "line": "10", "own_conf": "80%", "admin_desc": "", "end-user_desc": "", "gpt_conf": "", "snippet": ""},
        {"path": "a_file.py", "line": "5", "own_conf": "80%", "admin_desc": "", "end-user_desc": "", "gpt_conf": "", "snippet": ""},
    ]
    import_report.write_text(json.dumps(data))

    run_cli(
        targets=[],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        output_format="csv",
        quiet=True,
        import_file=str(import_report),
        sort_by="path"
    )

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
    assert len(lines) == 3
    assert "a_file.py" in lines[1]
    assert "z_file.py" in lines[2]


def test_main_cli_sort_by_parsing(monkeypatch, tmp_path):
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')\n")

    test_args = ["gptscan.py", str(test_file), "--cli", "--sort-by", "path", "--quiet"]
    monkeypatch.setattr(sys, "argv", test_args)

    with patch("gptscan.run_cli", return_value=0) as mock_run_cli:
        main()
        mock_run_cli.assert_called_once()
        _, kwargs = mock_run_cli.call_args
        assert kwargs.get("sort_by") == "path"
