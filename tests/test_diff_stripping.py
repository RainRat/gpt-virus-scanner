import pytest
from gptscan import unpack_content

def test_diff_prefix_stripping():
    """Verify that '+' and ' ' prefixes are stripped from Unified Diff hunks."""
    content = b"""--- a/file.py
+++ b/file.py
@@ -1,2 +1,3 @@
 def hello():
-    print("old")
+    print("new")
+    return True
"""
    results = list(unpack_content("test.diff", content))
    assert len(results) == 1
    _, snippet = results[0]

    # Expected: prefixes stripped, indentation preserved
    expected = b'def hello():\n    print("new")\n    return True'
    assert snippet == expected

def test_diff_prefix_stripping_with_mixed_indent():
    """Verify that indentation is correctly preserved after stripping prefixes."""
    content = b"""--- a/script.sh
+++ b/script.sh
@@ -10,3 +10,4 @@
 if [ "$1" = "test" ]; then
+    echo "Starting test..."
     ./run_test.sh
+    echo "Done."
 fi
"""
    results = list(unpack_content("test.diff", content))
    assert len(results) == 1
    _, snippet = results[0]

    expected = (
        b'if [ "$1" = "test" ]; then\n'
        b'    echo "Starting test..."\n'
        b'    ./run_test.sh\n'
        b'    echo "Done."\n'
        b'fi'
    )
    assert snippet == expected
