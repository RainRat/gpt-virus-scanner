import pytest
from gptscan import get_online_url
from unittest.mock import patch, MagicMock

def test_get_online_url_remote_target():
    # GitHub blob
    url = get_online_url("[URL] https://github.com/user/repo/blob/main/file.py", 10)
    assert url == "https://github.com/user/repo/blob/main/file.py#L10"

    # GitHub raw
    url = get_online_url("[URL] https://raw.githubusercontent.com/user/repo/main/file.py", 10)
    assert url == "https://github.com/user/repo/blob/main/file.py#L10"

    # GitLab raw
    url = get_online_url("[URL] https://gitlab.com/user/repo/-/raw/main/file.py", 10)
    assert url == "https://gitlab.com/user/repo/-/blob/main/file.py#L10"

    # Bitbucket
    url = get_online_url("[URL] https://bitbucket.org/user/repo/src/main/file.py", 10)
    assert url == "https://bitbucket.org/user/repo/src/main/file.py#lines-10"

@patch('gptscan._get_git_info')
@patch('gptscan._get_git_remote_url')
@patch('gptscan._get_git_revision')
def test_get_online_url_local_file(mock_rev, mock_remote, mock_info):
    mock_info.return_value = ("/app", "src/file.py")
    mock_remote.return_value = "git@github.com:user/repo.git"
    mock_rev.return_value = "abcdef"

    url = get_online_url("/app/src/file.py", 10)
    assert url == "https://github.com/user/repo/blob/abcdef/src/file.py#L10"

@patch('gptscan._get_git_info')
@patch('gptscan._get_git_remote_url')
def test_get_online_url_local_file_gitlab(mock_remote, mock_info):
    mock_info.return_value = ("/app", "src/file.py")
    mock_remote.return_value = "https://gitlab.com/user/repo.git"

    with patch('gptscan._get_git_revision', return_value="main"):
        url = get_online_url("/app/src/file.py", 10)
        assert url == "https://gitlab.com/user/repo/-/blob/main/src/file.py#L10"

@patch('gptscan._get_git_info')
@patch('gptscan._get_git_remote_url')
def test_get_online_url_unsupported_remote(mock_remote, mock_info):
    mock_info.return_value = ("/app", "src/file.py")
    mock_remote.return_value = "https://myserver.com/repo.git"

    url = get_online_url("/app/src/file.py", 10)
    assert url is None
