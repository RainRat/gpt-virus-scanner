import json
import pytest
import gptscan
from gptscan import load_report_file


def test_load_report_file_directory_recursive_and_formats(tmp_path):
    sub_dir = tmp_path / "reports"
    sub_dir.mkdir()
    nested_dir = sub_dir / "nested"
    nested_dir.mkdir()

    json_report = sub_dir / "report1.json"
    json_report.write_text(json.dumps([
        {
            "path": "app/main.py",
            "line": "10",
            "own_conf": "80%",
            "gpt_conf": "85%",
            "admin_desc": "Main entry point check",
            "end-user_desc": "User note",
            "snippet": "os.system(cmd)"
        }
    ]), encoding="utf-8")

    yaml_report = nested_dir / "report2.yaml"
    yaml_report.write_text(
        "- path: lib/utils.py\n"
        "  line: '25'\n"
        "  own_conf: 60%\n"
        "  gpt_conf: 70%\n"
        "  admin_desc: Utils check\n"
        "  end-user_desc: Safe\n"
        "  snippet: eval(input)\n",
        encoding="utf-8"
    )

    csv_report = nested_dir / "report3.csv"
    csv_report.write_text(
        "path,line,own_conf,gpt_conf,admin_desc,end-user_desc,snippet\n"
        "src/cli.py,5,90%,95%,CLI check,Warning,exec(arg)\n",
        encoding="utf-8"
    )

    results = load_report_file(str(sub_dir))

    assert len(results) == 3
    paths = {r["path"] for r in results}
    assert paths == {"app/main.py", "lib/utils.py", "src/cli.py"}


def test_load_report_file_directory_with_corrupted_and_empty_files(tmp_path):
    sub_dir = tmp_path / "mixed_reports"
    sub_dir.mkdir()

    valid_json = sub_dir / "valid.json"
    valid_json.write_text(json.dumps([
        {
            "path": "good.py",
            "line": "1",
            "own_conf": "50%",
            "gpt_conf": "50%",
            "admin_desc": "OK",
            "end-user_desc": "OK",
            "snippet": "print(1)"
        }
    ]), encoding="utf-8")

    corrupted_json = sub_dir / "corrupted.json"
    corrupted_json.write_text("{malformed json content: [", encoding="utf-8")

    empty_json = sub_dir / "empty.json"
    empty_json.write_text("  \n", encoding="utf-8")

    results = load_report_file(str(sub_dir))

    assert len(results) == 1
    assert results[0]["path"] == "good.py"


def test_load_report_file_directory_unsupported_extensions(tmp_path):
    sub_dir = tmp_path / "unsupported_dir"
    sub_dir.mkdir()

    binary_file = sub_dir / "data.bin"
    binary_file.write_bytes(b"\x00\x01\x02\x03")

    image_file = sub_dir / "photo.png"
    image_file.write_bytes(b"\x89PNG\r\n\x1a\n")

    results = load_report_file(str(sub_dir))
    assert results == []


def test_load_report_file_directory_empty(tmp_path):
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    results = load_report_file(str(empty_dir))
    assert results == []


def test_load_report_file_single_empty_file_raises(tmp_path):
    empty_file = tmp_path / "empty_report.json"
    empty_file.write_text("\n\t  \n", encoding="utf-8")

    with pytest.raises(ValueError, match="File is empty."):
        load_report_file(str(empty_file))
