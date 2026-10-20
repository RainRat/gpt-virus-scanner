import pytest
import gptscan

def test_generate_csv_default_delimiter_and_fields():
    results = [
        {
            "path": "test1.py",
            "line": 10,
            "own_conf": "80%",
            "gpt_conf": "90%",
            "admin_desc": "Admin note",
            "end-user_desc": "User note",
            "snippet": "os.system('rm -rf /')",
            "extra_field": "ignored"
        }
    ]
    csv_out = gptscan.generate_csv(results)
    lines = csv_out.strip().split("\r\n") if "\r\n" in csv_out else csv_out.strip().split("\n")
    assert lines[0] == "path,line,own_conf,gpt_conf,admin_desc,end-user_desc,snippet"
    assert "test1.py" in lines[1]
    assert "80%" in lines[1]
    assert "ignored" not in lines[1]

def test_generate_csv_custom_delimiter_and_fields():
    results = [
        {
            "path": "test2.js",
            "own_conf": "50%",
            "line": 5
        }
    ]
    fields = ["path", "line", "own_conf"]
    tsv_out = gptscan.generate_csv(results, delimiter='\t', fields=fields)
    lines = tsv_out.strip().split("\r\n") if "\r\n" in tsv_out else tsv_out.strip().split("\n")
    assert lines[0] == "path\tline\town_conf"
    assert lines[1] == "test2.js\t5\t50%"
