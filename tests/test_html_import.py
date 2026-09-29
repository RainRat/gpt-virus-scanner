import os
import pytest
import gptscan


def test_parse_html_content_roundtrip():
    results = [
        {
            "path": "scripts/malware.py",
            "line": 42,
            "own_conf": "95%",
            "gpt_conf": "90%",
            "admin_desc": "Suspicious execution of subprocess with shell=True & dangerous flags",
            "end-user_desc": "This script may execute arbitrary shell commands.",
            "snippet": "import subprocess\nsubprocess.run('rm -rf /', shell=True)",
        }
    ]

    html_report = gptscan.generate_html(results)
    parsed = gptscan.parse_html_content(html_report)

    assert len(parsed) == 1
    item = parsed[0]
    assert item["path"] == "scripts/malware.py"
    assert item["line"] == "42"
    assert item["own_conf"] == "90%"  # effective threat level from generate_html
    assert "subprocess" in item["admin_desc"]
    assert "shell commands" in item["end-user_desc"]
    assert "rm -rf /" in item["snippet"]


def test_parse_html_content_escaped_entities_and_newlines():
    raw_html = """
    <!DOCTYPE html>
    <html>
    <body>
        <table>
            <tr>
                <th>Path</th><th>Line</th><th>Threat Level</th><th>Analysis</th><th>Links</th><th>Snippet</th>
            </tr>
            <tr>
                <td>app/&lt;config&gt;.py</td>
                <td>15</td>
                <td>85%</td>
                <td>
                    <strong>Admin:</strong> Line 1 &lt;br&gt; Line 2<br>
                    <strong>User:</strong> User note &amp; details
                </td>
                <td><a href="#">Link</a></td>
                <td><pre><code>if x &lt; 10:\n    print(&quot;danger&quot;)</code></pre></td>
            </tr>
        </table>
    </body>
    </html>
    """

    parsed = gptscan.parse_html_content(raw_html)
    assert len(parsed) == 1
    item = parsed[0]
    assert item["path"] == "app/<config>.py"
    assert item["line"] == "15"
    assert item["own_conf"] == "85%"
    assert item["admin_desc"] == "Line 1 <br> Line 2"
    assert item["end-user_desc"] == "User note & details"
    assert item["snippet"] == 'if x < 10:\n    print("danger")'


def test_parse_html_content_empty_or_no_rows():
    assert gptscan.parse_html_content("") == []
    assert gptscan.parse_html_content("   ") == []

    no_table_html = "<html><body><h1>No results</h1></body></html>"
    assert gptscan.parse_html_content(no_table_html) == []


def test_parse_report_content_html_autodetect():
    results = [
        {
            "path": "test.js",
            "line": 1,
            "own_conf": "70%",
            "admin_desc": "Eval usage",
            "end-user_desc": "Avoid eval",
            "snippet": "eval('alert(1)')",
        }
    ]

    html_content = gptscan.generate_html(results)

    # Autodetect by content structure (starts with < and contains <table>)
    parsed = gptscan.parse_report_content(html_content)
    assert len(parsed) == 1
    assert parsed[0]["path"] == "test.js"

    # Auto-detect via filename_hint .html
    parsed_hint = gptscan.parse_report_content(html_content, filename_hint="report.html")
    assert len(parsed_hint) == 1
    assert parsed_hint[0]["path"] == "test.js"

    # Auto-detect via filename_hint .htm / .xhtml
    parsed_htm = gptscan.parse_report_content(html_content, filename_hint="report.htm")
    assert len(parsed_htm) == 1

    parsed_xhtml = gptscan.parse_report_content(html_content, filename_hint="report.xhtml")
    assert len(parsed_xhtml) == 1
