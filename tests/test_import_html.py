import pytest
import html
from gptscan import parse_report_content, generate_html

def test_html_import_roundtrip():
    """Verify that results exported as HTML can be imported back correctly."""
    results = [
        {
            "path": "test1.py",
            "line": 10,
            "own_conf": "80%",
            "gpt_conf": "90%",
            "admin_desc": "Technical analysis here.",
            "end-user_desc": "User friendly note.",
            "snippet": "import os\nos.system('rm -rf /')"
        },
        {
            "path": "test2.js",
            "line": "-",
            "own_conf": "45%",
            "gpt_conf": "",
            "admin_desc": "",
            "end-user_desc": "",
            "snippet": "console.log('safe')"
        }
    ]

    # Generate HTML
    html_content = generate_html(results)

    # Parse HTML back
    imported_results = parse_report_content(html_content, filename_hint="report.html")

    assert len(imported_results) == 2

    # Check first result (with AI data)
    r1 = imported_results[0]
    assert r1["path"] == "test1.py"
    assert r1["line"] == "10"
    # generate_html uses (gpt_conf or own_conf) for Confidence column
    # Confidence in HTML will be "90%"
    assert r1["own_conf"] == "90%"
    assert r1["admin_desc"] == "Technical analysis here."
    assert r1["end-user_desc"] == "User friendly note."
    assert r1["snippet"] == "import os\nos.system('rm -rf /')"

    # Check second result (without AI data)
    r2 = imported_results[1]
    assert r2["path"] == "test2.js"
    assert r2["line"] == "-"
    assert r2["own_conf"] == "45%"
    assert r2.get("admin_desc", "") == ""
    assert r2.get("end-user_desc", "") == ""
    assert r2["snippet"] == "console.log('safe')"

def test_html_import_entities():
    """Verify that HTML entities are correctly unescaped during import."""
    html_content = """
    <table>
        <tr><th>Path</th><th>Line</th><th>Confidence</th><th>Analysis</th><th>Snippet</th></tr>
        <tr>
            <td>path&lt;script&gt;.py</td>
            <td>1</td>
            <td>50%</td>
            <td><strong>Admin:</strong> notes &amp; details<br><strong>User:</strong> user &quot;note&quot;</td>
            <td><pre><code>x &gt; 1 &amp;&amp; y &lt; 2</code></pre></td>
        </tr>
    </table>
    """
    imported = parse_report_content(html_content, filename_hint="report.html")
    assert len(imported) == 1
    r = imported[0]
    assert r["path"] == "path<script>.py"
    assert r["admin_desc"] == "notes & details"
    assert r["end-user_desc"] == 'user "note"'
    assert r["snippet"] == "x > 1 && y < 2"

def test_html_import_no_ai_notes():
    """Verify import when Analysis column doesn't have Admin/User labels."""
    html_content = """
    <table>
        <tr>
            <td>plain.py</td>
            <td>5</td>
            <td>30%</td>
            <td>Simple analysis text without labels.</td>
            <td><pre><code>code</code></pre></td>
        </tr>
    </table>
    """
    imported = parse_report_content(html_content, filename_hint="report.html")
    assert len(imported) == 1
    r = imported[0]
    assert r["path"] == "plain.py"
    # If no Admin/User labels, they stay empty.
    assert r.get("admin_desc", "") == ""
    assert r.get("end-user_desc", "") == ""
