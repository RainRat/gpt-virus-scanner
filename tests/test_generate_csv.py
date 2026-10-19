from gptscan import generate_csv


def test_generate_csv_default_delimiter():
    results = [
        {
            "path": "app/main.py",
            "line": "10",
            "own_conf": "80%",
            "gpt_conf": "85%",
            "admin_desc": "Main entry point check",
            "end-user_desc": "User note",
            "snippet": "os.system(cmd)"
        }
    ]
    output = generate_csv(results)
    lines = [line.strip() for line in output.strip().splitlines()]
    assert len(lines) == 2
    assert lines[0] == "path,line,own_conf,gpt_conf,admin_desc,end-user_desc,snippet"
    assert "app/main.py,10,80%,85%,Main entry point check,User note,os.system(cmd)" in lines[1]


def test_generate_csv_tsv_delimiter():
    results = [
        {
            "path": "lib/utils.py",
            "line": "25",
            "own_conf": "60%",
            "gpt_conf": "70%",
            "admin_desc": "Utils check",
            "end-user_desc": "Safe",
            "snippet": "eval(input)"
        }
    ]
    output = generate_csv(results, delimiter="\t")
    lines = [line.strip() for line in output.strip().splitlines()]
    assert len(lines) == 2
    assert lines[0] == "path\tline\town_conf\tgpt_conf\tadmin_desc\tend-user_desc\tsnippet"
    assert "lib/utils.py\t25\t60%\t70%\tUtils check\tSafe\teval(input)" in lines[1]


def test_generate_csv_empty_results():
    output = generate_csv([])
    lines = [line.strip() for line in output.strip().splitlines()]
    assert len(lines) == 1
    assert lines[0] == "path,line,own_conf,gpt_conf,admin_desc,end-user_desc,snippet"
