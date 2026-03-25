import pytest
import gptscan
from gptscan import unpack_content

def test_unpack_content_html_extraction():
    """Verify that <script> blocks are correctly extracted from HTML files."""
    html_content = """
    <html>
        <head>
            <script>
                console.log('Script 1');
            </script>
        </head>
        <body>
            <h1>Test</h1>
            <script type="text/javascript">
                const x = 10;
                alert(x);
            </script>
            <script>
                // Empty script should be ignored if strip() is empty
            </script>
            <script>    </script>
            <p>More text</p>
            <SCRIPT>
                console.log('Case insensitive');
            </SCRIPT>
        </body>
    </html>
    """
    content_bytes = html_content.encode('utf-8')

    # Test with .html extension
    snippets = list(unpack_content("test.html", content_bytes))

    assert len(snippets) == 4

    assert snippets[0][0] == "test.html [Script 1]"
    assert "console.log('Script 1');" in snippets[0][1].decode('utf-8')

    assert snippets[1][0] == "test.html [Script 2]"
    assert "const x = 10;" in snippets[1][1].decode('utf-8')
    assert "alert(x);" in snippets[1][1].decode('utf-8')

    assert snippets[2][0] == "test.html [Script 3]"
    assert "// Empty script" in snippets[2][1].decode('utf-8')

    assert snippets[3][0] == "test.html [Script 5]"
    assert "console.log('Case insensitive');" in snippets[3][1].decode('utf-8')

def test_unpack_content_html_no_scripts():
    """Verify that HTML files with no scripts yield no snippets."""
    html_content = "<html><body><h1>No Scripts</h1></body></html>"
    content_bytes = html_content.encode('utf-8')

    snippets = list(unpack_content("noscripts.html", content_bytes))
    assert len(snippets) == 0

def test_unpack_content_xhtml_extraction():
    """Verify that <script> blocks are correctly extracted from XHTML files."""
    xhtml_content = """
    <html xmlns="http://www.w3.org/1999/xhtml">
        <script>alert('XHTML');</script>
    </html>
    """
    snippets = list(unpack_content("test.xhtml", xhtml_content.encode('utf-8')))
    assert len(snippets) == 1
    assert snippets[0][0] == "test.xhtml [Script 1]"
    assert b"alert('XHTML');" in snippets[0][1]
