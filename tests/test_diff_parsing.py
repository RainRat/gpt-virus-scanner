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
    assert b"print(\"hello\")\nimport os; os.system(\"evil\")" == snippet

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
    assert b"new" == results[0][1]
    assert "test.diff [b.js @ line 5]" == results[1][0]
    assert b"eval(\"malicious\")\ncontext" == results[1][1]

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
    assert b"#!/bin/bash\nrm -rf /" == results[0][1]

def test_unpack_diff_complex_header():
    content = b"""--- old/file.py\t2023-01-01 12:00:00.000000000 +0000
+++ new/file.py\t2023-01-01 12:01:00.000000000 +0000
@@ -42,1 +42,1 @@
+suspicious_call()
"""
    results = list(unpack_content("patch.patch", content))
    assert len(results) == 1
    assert "patch.patch [new/file.py @ line 42]" == results[0][0]
    assert b"suspicious_call()" == results[0][1]

def test_unpack_diff_index_and_commit_prefixes():
    index_content = b"Index: app.py\n--- app.py\n+++ app.py\n@@ -1,1 +1,1 @@\n+exec('secret')"
    index_results = list(unpack_content("svn_patch.txt", index_content))
    assert len(index_results) == 1
    assert index_results[0][0] == "svn_patch.txt [app.py @ line 1]"
    assert b"exec('secret')" in index_results[0][1]

    commit_content = b"commit 1234567890abcdef\n--- server.py\n+++ server.py\n@@ -10,1 +10,1 @@\n+os.system('sh')"
    commit_results = list(unpack_content("git_commit.txt", commit_content))
    assert len(commit_results) == 1
    assert commit_results[0][0] == "git_commit.txt [server.py @ line 10]"
    assert b"os.system('sh')" in commit_results[0][1]

def test_unpack_diff_malformed_hunk_header():
    content = b"--- a/config.py\n+++ b/config.py\n@@ malformed @@\n+DEBUG = True"
    results = list(unpack_content("patch.diff", content))
    assert len(results) == 1
    assert results[0][0] == "patch.diff [config.py @ unknown]"
    assert b"DEBUG = True" in results[0][1]

def test_unpack_diff_non_diff_line_reset():
    content = b"--- a/main.py\n+++ b/main.py\n@@ -1,1 +1,1 @@\n+x = 1\nrandom non-diff comment line\n+y = 2"
    results = list(unpack_content("patch.diff", content))
    assert len(results) == 1
    assert results[0][0] == "patch.diff [main.py @ line 1]"
    assert results[0][1] == b"x = 1"

def test_unpack_diff_interrupted_by_dash_header():
    content = b"--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n+added line\n--- a/b.py\n+++ b/b.py\n@@ -2,1 +2,1 @@\n+second line"
    results = list(unpack_content("patch.diff", content))
    assert len(results) == 2
    assert results[0][0] == "patch.diff [a.py @ line 1]"
    assert results[0][1] == b"added line"
    assert results[1][0] == "patch.diff [b.py @ line 2]"
    assert results[1][1] == b"second line"

def test_unpack_diff_without_ab_prefix():
    content = b"--- src/utils.py\n+++ src/utils.py\n@@ -5,1 +5,1 @@\n+validate()"
    results = list(unpack_content("patch.diff", content))
    assert len(results) == 1
    assert results[0][0] == "patch.diff [src/utils.py @ line 5]"
    assert results[0][1] == b"validate()"
