import html
from gptscan import generate_console_report

def test_console_report_html_decoding():
    results = [
        {
            "path": "test&amp;file.py",
            "own_conf": "90%",
            "gpt_conf": "95%",
            "admin_desc": "Dangerous &lt;script&gt; found",
            "end-user_desc": "Suspicious code",
            "line": "10",
            "snippet": "os.system('rm -rf /') &amp;&amp; echo 'done'"
        }
    ]
    report = generate_console_report(results, use_color=False)
    assert "test&file.py" in report
    assert "Dangerous <script> found" in report
    assert "os.system('rm -rf /') && echo 'done'" in report
    assert "&amp;" not in report
    assert "&lt;" not in report

def test_console_report_ansi_location_bolding():
    results = [
        {
            "path": "test.py",
            "own_conf": "90%",
            "line": "10",
            "snippet": "print('hello')"
        }
    ]
    # Check for the sequence: RISK_LABEL_RESET BOLD - LOCATION
    # \033[0m \033[1m- test.py:10
    report = generate_console_report(results, use_color=True)
    # Finding 1 is high risk (RED)
    assert "\x1b[0m \x1b[1m- test.py:10\x1b[0m" in report

def test_console_report_threat_score_coloring():
    results = [
        {
            "path": "test.py",
            "own_conf": "90%",
            "gpt_conf": "40%",
            "line": "10",
            "snippet": "print('hello')"
        }
    ]
    report = generate_console_report(results, use_color=True)
    # 90% should be RED (\x1b[1;91m) and NO extra BOLD (\x1b[1m)
    assert "\x1b[1;91m90%\x1b[0m" in report
    assert "\x1b[1;91m\x1b[1m90%\x1b[0m" not in report

    # 40% should NOT be bolded or colored
    assert "AI:\x1b[0m 40%" in report
