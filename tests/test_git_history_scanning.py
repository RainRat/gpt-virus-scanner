import os
import subprocess
import pytest
from pathlib import Path
from gptscan import get_git_history_snippets, unpack_content

def test_get_git_history_snippets(tmp_path):
    # Initialize a git repo
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "you@example.com"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Your Name"], cwd=repo_dir, check=True)

    # Create a file and commit it
    test_file = repo_dir / "test.py"
    test_file.write_text("print('initial')")
    subprocess.run(["git", "add", "test.py"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=repo_dir, check=True)

    # Modify and commit again
    test_file.write_text("print('modified')")
    subprocess.run(["git", "add", "test.py"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "second commit"], cwd=repo_dir, check=True)

    # Get snippets (default count is 5)
    snippets = get_git_history_snippets(str(repo_dir), count=2)

    assert len(snippets) == 2
    # Commits are in reverse chronological order
    name1, content1 = snippets[0]
    assert "second commit" in name1
    assert b"commit " in content1
    assert b"print('modified')" in content1

    name2, content2 = snippets[1]
    assert "initial commit" in name2
    assert b"commit " in content2
    assert b"print('initial')" in content2

def test_unpack_git_commit(tmp_path):
    # Test that unpack_content correctly handles git commit format
    commit_content = b"""commit abcdef1234567890
Author: Test Author <test@example.com>
Date:   Mon Jan 1 00:00:00 2025 +0000

    test commit

diff --git a/test.py b/test.py
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/test.py
@@ -0,0 +1,1 @@
+import os; os.system('rm -rf /')
"""
    results = list(unpack_content("test_commit", commit_content))
    assert len(results) >= 1
    name, content = results[0]
    assert "test.py" in name
    assert b"import os; os.system('rm -rf /')" in content

def test_get_git_history_snippets_non_git(tmp_path):
    non_git_dir = tmp_path / "non_git"
    non_git_dir.mkdir()

    # Get snippets
    snippets = get_git_history_snippets(str(non_git_dir))
    assert snippets == []
