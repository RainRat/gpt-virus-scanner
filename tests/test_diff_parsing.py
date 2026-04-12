import pytest
from gptscan import unpack_content

def test_unpack_diff_single_file():
    content = b"""--- a/file.py
+++ b/file.py
@@ -1,1 +1,2 @@
 print("hello")
+import os; os.system("evil")
"""
    results = list(unpack_content("test.diff", content))
    assert len(results) == 1
    name, snippet = results[0]
    assert "test.diff [file.py @ line 1]" == name
    assert b" print(\"hello\")\n+import os; os.system(\"evil\")" == snippet

def test_unpack_diff_multi_file():
    content = b"""--- a/a.py
+++ b/a.py
@@ -10,1 +10,1 @@
-old
+new
--- a/b.js
+++ b/b.js
@@ -5,1 +5,2 @@
+eval("malicious")
 context
"""
    results = list(unpack_content("test.diff", content))
    assert len(results) == 2
    assert "test.diff [a.py @ line 10]" == results[0][0]
    assert b"+new" == results[0][1]
    assert "test.diff [b.js @ line 5]" == results[1][0]
    assert b"+eval(\"malicious\")\n context" == results[1][1]

def test_unpack_diff_no_additions():
    content = b"""--- a/file.py
+++ b/file.py
@@ -1,1 +1,0 @@
-print("goodbye")
"""
    results = list(unpack_content("test.diff", content))
    assert len(results) == 0

def test_unpack_diff_added_file():
    content = b"""--- /dev/null
+++ b/new_script.sh
@@ -0,0 +1,1 @@
+#!/bin/bash
+rm -rf /
"""
    results = list(unpack_content("test.diff", content))
    assert len(results) == 1
    assert "test.diff [new_script.sh @ line 1]" == results[0][0]
    assert b"+#!/bin/bash\n+rm -rf /" == results[0][1]

def test_unpack_diff_complex_header():
    # Diff with tabs and timestamps in header
    content = b"""--- old/file.py\t2023-01-01 12:00:00.000000000 +0000
+++ new/file.py\t2023-01-01 12:01:00.000000000 +0000
@@ -42,1 +42,1 @@
+suspicious_call()
"""
    results = list(unpack_content("patch.patch", content))
    assert len(results) == 1
    assert "patch.patch [new/file.py @ line 42]" == results[0][0]
    assert b"+suspicious_call()" == results[0][1]
