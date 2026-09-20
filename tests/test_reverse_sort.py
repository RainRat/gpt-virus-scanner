import json
import pytest
import gptscan

def test_run_cli_reverse_sort_path(capsys, monkeypatch):
    """Test --reverse flag with --sort-by path."""
    content = json.dumps([
        {"path": "a.py", "own_conf": "50%", "line": "1", "snippet": "code", "admin_desc": "", "end-user_desc": "", "gpt_conf": ""},
        {"path": "c.py", "own_conf": "50%", "line": "1", "snippet": "code", "admin_desc": "", "end-user_desc": "", "gpt_conf": ""},
        {"path": "b.py", "own_conf": "50%", "line": "1", "snippet": "code", "admin_desc": "", "end-user_desc": "", "gpt_conf": ""},
    ])

    monkeypatch.setattr('sys.stdin', type('Dummy', (), {'read': lambda self: content})())

    gptscan.run_cli(
        targets=[],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_format='json',
        import_file='-',
        sort_by='path',
        reverse_sort=True,
        quiet=True
    )

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
    paths = [json.loads(line)['path'] for line in lines]
    assert paths == ["c.py", "b.py", "a.py"]

def test_run_cli_reverse_sort_line(capsys, monkeypatch):
    """Test --reverse flag with --sort-by line."""
    content = json.dumps([
        {"path": "file1.py", "own_conf": "50%", "line": "4", "snippet": "code", "admin_desc": "", "end-user_desc": "", "gpt_conf": ""},
        {"path": "file2.py", "own_conf": "50%", "line": "1", "snippet": "code", "admin_desc": "", "end-user_desc": "", "gpt_conf": ""},
        {"path": "file3.py", "own_conf": "50%", "line": "2", "snippet": "code", "admin_desc": "", "end-user_desc": "", "gpt_conf": ""},
    ])

    monkeypatch.setattr('sys.stdin', type('Dummy', (), {'read': lambda self: content})())

    gptscan.run_cli(
        targets=[],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_format='json',
        import_file='-',
        sort_by='line',
        reverse_sort=True,
        quiet=True
    )

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
    line_nums = [int(json.loads(line)['line']) for line in lines]
    assert line_nums == [4, 2, 1]

def test_run_cli_reverse_sort_threat(capsys, monkeypatch):
    """Test --reverse flag with --sort-by threat."""
    content = json.dumps([
        {"path": "low.py", "own_conf": "20%", "line": "1", "snippet": "code", "admin_desc": "", "end-user_desc": "", "gpt_conf": ""},
        {"path": "high.py", "own_conf": "90%", "line": "1", "snippet": "code", "admin_desc": "", "end-user_desc": "", "gpt_conf": ""},
        {"path": "med.py", "own_conf": "50%", "line": "1", "snippet": "code", "admin_desc": "", "end-user_desc": "", "gpt_conf": ""},
    ])

    monkeypatch.setattr('sys.stdin', type('Dummy', (), {'read': lambda self: content})())

    gptscan.run_cli(
        targets=[],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_format='json',
        import_file='-',
        sort_by='threat',
        reverse_sort=True,
        quiet=True
    )

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
    paths = [json.loads(line)['path'] for line in lines]
    assert paths == ["low.py", "med.py", "high.py"]

def test_run_cli_reverse_sort_default_threat(capsys, monkeypatch):
    """Test --reverse flag with default threat sorting when sort_by is None."""
    content = json.dumps([
        {"path": "high.py", "own_conf": "90%", "line": "1", "snippet": "code", "admin_desc": "", "end-user_desc": "", "gpt_conf": ""},
        {"path": "low.py", "own_conf": "20%", "line": "1", "snippet": "code", "admin_desc": "", "end-user_desc": "", "gpt_conf": ""},
        {"path": "med.py", "own_conf": "50%", "line": "1", "snippet": "code", "admin_desc": "", "end-user_desc": "", "gpt_conf": ""},
    ])

    monkeypatch.setattr('sys.stdin', type('Dummy', (), {'read': lambda self: content})())

    gptscan.run_cli(
        targets=[],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_format='json',
        import_file='-',
        reverse_sort=True,
        quiet=True
    )

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
    paths = [json.loads(line)['path'] for line in lines]
    assert paths == ["low.py", "med.py", "high.py"]

def test_main_reverse_flag_parsing(capsys, monkeypatch):
    """Test CLI argument parsing and execution with -r / --reverse in main()."""
    content = json.dumps([
        {"path": "x.py", "own_conf": "80%", "line": "1", "snippet": "code", "admin_desc": "", "end-user_desc": "", "gpt_conf": ""},
        {"path": "y.py", "own_conf": "30%", "line": "1", "snippet": "code", "admin_desc": "", "end-user_desc": "", "gpt_conf": ""},
    ])

    monkeypatch.setattr('sys.argv', ['gptscan.py', '--import-results', '-', '--json', '--sort-by', 'path', '-r', '--cli', '-q'])
    monkeypatch.setattr('sys.stdin', type('Dummy', (), {'read': lambda self: content})())

    gptscan.main()

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
    paths = [json.loads(line)['path'] for line in lines]
    assert paths == ["y.py", "x.py"]
