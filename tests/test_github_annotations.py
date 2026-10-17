import io
import json
import pytest
import gptscan


def test_generate_github_annotations_basic():
    results = [
        {
            "path": "src/danger.py",
            "line": "15",
            "own_conf": "90%",
            "gpt_conf": "95%",
            "admin_desc": "Critical command execution vulnerability.",
            "end-user_desc": "Do not run.",
            "snippet": "import os\nos.system('rm -rf /')"
        },
        {
            "path": "src/warn.py",
            "line": "3",
            "own_conf": "60%",
            "gpt_conf": "",
            "admin_desc": "Suspicious eval usage.",
            "end-user_desc": "",
            "snippet": "eval(user_input)"
        }
    ]

    output = gptscan.generate_github_annotations(results)
    lines = output.splitlines()

    assert len(lines) == 2
    assert lines[0].startswith("::error file=src/danger.py,line=15,title=Threat Level 95%25::Admin: Critical command execution vulnerability. | User: Do not run. | Snippet: import os")
    assert lines[1].startswith("::warning file=src/warn.py,line=3,title=Threat Level 60%25::Admin: Suspicious eval usage. | Snippet: eval(user_input)")


def test_generate_github_annotations_escaping():
    results = [
        {
            "path": "src/multiline.py",
            "line": "10",
            "own_conf": "85%",
            "gpt_conf": "",
            "admin_desc": "Line 1\nLine 2 % 100",
            "end-user_desc": "",
            "snippet": "print('hello')"
        }
    ]

    output = gptscan.generate_github_annotations(results)
    assert "%25" in output
    assert "%0A" in output
    assert "Line 1%0ALine 2 %25 100" in output


def test_parse_github_annotations_content():
    content = (
        "::error file=src/danger.py,line=15,title=Threat Level 90%25::Admin: Critical command execution%0ASecond line\n"
        "::warning file=src/warn.py,line=5,title=Threat Level 65%25::Suspicious code detected\n"
    )

    parsed = gptscan.parse_github_annotations_content(content)
    assert len(parsed) == 2

    assert parsed[0]["path"] == "src/danger.py"
    assert parsed[0]["line"] == "15"
    assert parsed[0]["own_conf"] == "90%"
    assert "Second line" in parsed[0]["admin_desc"]

    assert parsed[1]["path"] == "src/warn.py"
    assert parsed[1]["line"] == "5"
    assert parsed[1]["own_conf"] == "65%"


def test_parse_report_content_auto_detect_github_annotations():
    content = "::error file=app.py,line=1,title=Threat Level 80%25::Suspicious code"
    results = gptscan.parse_report_content(content)

    assert len(results) == 1
    assert results[0]["path"] == "app.py"
    assert results[0]["line"] == "1"
    assert results[0]["own_conf"] == "80%"


def test_export_results_to_file_github_annotations(tmp_path):
    results = [
        {
            "path": "test.py",
            "line": "12",
            "own_conf": "85%",
            "gpt_conf": "",
            "admin_desc": "High risk",
            "end-user_desc": "",
            "snippet": "code()"
        }
    ]

    out_file = tmp_path / "results.annotations"
    gptscan.export_results_to_file(str(out_file), results, output_format='github')

    content = out_file.read_text(encoding='utf-8')
    assert "::error file=test.py,line=12,title=Threat Level 85%25::" in content


def test_cli_github_annotations_flag(tmp_path):
    out_file = tmp_path / "out.annotations"
    # Test run_cli with github output format and output_file
    gptscan.run_cli(
        [],
        deep=False,
        show_all=False,
        use_gpt=False,
        rate_limit=60,
        output_format='github',
        output_file=str(out_file),
        extra_snippets=[("script.py", b"import subprocess\nsubprocess.run(cmd)")],
    )

    out = out_file.read_text(encoding='utf-8')
    assert "::error file=script.py" in out or "::warning file=script.py" in out


def test_copy_as_github_annotations_gui(monkeypatch):
    class MockTree:
        def selection(self):
            return ["item1"]

    class MockClipboard:
        def __init__(self):
            self.content = ""
        def clear(self):
            self.content = ""
        def append(self, text):
            self.content += text

    clip = MockClipboard()
    mock_tree = MockTree()

    monkeypatch.setattr(gptscan, "tree", mock_tree)
    monkeypatch.setattr(gptscan, "_get_tree_results_as_dicts", lambda selection: [{
        "path": "clip.py",
        "line": "2",
        "own_conf": "80%",
        "gpt_conf": "",
        "admin_desc": "Clip test",
        "end-user_desc": "",
        "snippet": "test()"
    }])

    # Patch clipboard on tree
    mock_tree.clipboard_clear = clip.clear
    mock_tree.clipboard_append = clip.append
    monkeypatch.setattr(gptscan, "update_status", lambda msg: None)

    gptscan.copy_as_github_annotations()
    assert "::error file=clip.py,line=2,title=Threat Level 80%25::" in clip.content
