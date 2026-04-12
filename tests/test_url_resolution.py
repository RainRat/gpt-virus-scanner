import pytest
from gptscan import resolve_remote_url

def test_resolve_github_blob():
    url = "https://github.com/user/repo/blob/main/script.py"
    expected = "https://raw.githubusercontent.com/user/repo/main/script.py"
    assert resolve_remote_url(url) == expected

def test_resolve_github_gist():
    url = "https://gist.github.com/user/1234567890abcdef"
    expected = "https://gist.github.com/user/1234567890abcdef/raw"
    assert resolve_remote_url(url) == expected

def test_resolve_github_repo():
    url = "https://github.com/user/repo"
    expected = "https://github.com/user/repo/archive/HEAD.zip"
    assert resolve_remote_url(url) == expected

def test_resolve_github_repo_with_trailing_slash():
    url = "https://github.com/user/repo/"
    expected = "https://github.com/user/repo/archive/HEAD.zip"
    assert resolve_remote_url(url) == expected

def test_resolve_github_special_pages_not_repo():
    special_pages = [
        "https://github.com/user/repo/issues",
        "https://github.com/user/repo/pulls",
        "https://github.com/user/repo/actions",
        "https://github.com/user/repo/projects",
        "https://github.com/user/repo/wiki",
        "https://github.com/user/repo/security",
        "https://github.com/user/repo/insights",
        "https://github.com/user/repo/settings"
    ]
    for url in special_pages:
        assert resolve_remote_url(url) == url

def test_resolve_gitlab_blob():
    url = "https://gitlab.com/user/repo/-/blob/main/script.py"
    expected = "https://gitlab.com/user/repo/-/raw/main/script.py"
    assert resolve_remote_url(url) == expected

def test_resolve_standard_url():
    url = "https://example.com/script.sh"
    assert resolve_remote_url(url) == url

def test_resolve_non_url():
    path = "/home/user/script.py"
    assert resolve_remote_url(path) == path

    path_relative = "./script.py"
    assert resolve_remote_url(path_relative) == path_relative

def test_resolve_url_with_fragment():
    url = "https://github.com/user/repo/blob/main/script.py#L10"
    expected = "https://raw.githubusercontent.com/user/repo/main/script.py"
    assert resolve_remote_url(url) == expected

def test_resolve_github_case_insensitivity():
    url = "HTTPS://GITHUB.COM/user/repo/BLOB/main/script.py"
    expected = "https://raw.githubusercontent.com/user/repo/main/script.py"
    assert resolve_remote_url(url) == expected

def test_resolve_gitlab_repo():
    url = "https://gitlab.com/user/repo"
    expected = "https://gitlab.com/user/repo/-/archive/main/repo-main.zip"
    assert resolve_remote_url(url) == expected

def test_resolve_github_pull_request():
    url = "https://github.com/user/repo/pull/123"
    expected = "https://github.com/user/repo/pull/123.diff"
    assert resolve_remote_url(url) == expected

def test_resolve_github_commit():
    url = "https://github.com/user/repo/commit/abc123def"
    expected = "https://github.com/user/repo/commit/abc123def.diff"
    assert resolve_remote_url(url) == expected

def test_resolve_github_branch():
    url = "https://github.com/user/repo/tree/feature-branch"
    expected = "https://github.com/user/repo/archive/refs/heads/feature-branch.zip"
    assert resolve_remote_url(url) == expected

def test_resolve_gitlab_merge_request():
    url = "https://gitlab.com/user/repo/-/merge_requests/456"
    expected = "https://gitlab.com/user/repo/-/merge_requests/456.diff"
    assert resolve_remote_url(url) == expected

def test_resolve_bitbucket_repo():
    url = "https://bitbucket.org/user/repo"
    expected = "https://bitbucket.org/user/repo/get/HEAD.zip"
    assert resolve_remote_url(url) == expected

def test_resolve_bitbucket_raw():
    url = "https://bitbucket.org/user/repo/src/main/script.py"
    expected = "https://bitbucket.org/user/repo/raw/main/script.py"
    assert resolve_remote_url(url) == expected

def test_resolve_github_www_prefix():
    url = "https://www.github.com/user/repo/blob/main/script.py"
    expected = "https://raw.githubusercontent.com/user/repo/main/script.py"
    assert resolve_remote_url(url) == expected
