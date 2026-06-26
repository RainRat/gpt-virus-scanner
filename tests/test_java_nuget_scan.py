import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from gptscan import (
    get_java_packages_paths,
    get_dotnet_packages_paths,
    get_system_audit_data
)

def test_get_java_packages_paths(tmp_path):
    home = tmp_path / "home"
    home.mkdir()

    m2_repo = home / ".m2" / "repository"
    m2_repo.mkdir(parents=True)

    gradle_cache = home / ".gradle" / "caches" / "modules-2" / "files-2.1"
    gradle_cache.mkdir(parents=True)

    with patch("pathlib.Path.home", return_value=home):
        paths = get_java_packages_paths()
        assert str(m2_repo) in paths
        assert str(gradle_cache) in paths

def test_get_dotnet_packages_paths(tmp_path):
    home = tmp_path / "home"
    home.mkdir()

    nuget_repo = home / ".nuget" / "packages"
    nuget_repo.mkdir(parents=True)

    # Test default
    with patch("pathlib.Path.home", return_value=home), \
         patch.dict(os.environ, {}, clear=True):
        paths = get_dotnet_packages_paths()
        assert str(nuget_repo) in paths

    # Test environment variable
    custom_nuget = tmp_path / "custom_nuget"
    custom_nuget.mkdir()
    with patch("pathlib.Path.home", return_value=home), \
         patch.dict(os.environ, {"NUGET_PACKAGES": str(custom_nuget)}):
        paths = get_dotnet_packages_paths()
        assert str(custom_nuget) in paths
        assert str(nuget_repo) in paths

def test_system_audit_integration():
    with patch("gptscan.get_java_packages_paths", return_value=["/fake/java"]), \
         patch("gptscan.get_dotnet_packages_paths", return_value=["/fake/dotnet"]):
        paths, _ = get_system_audit_data()
        assert "/fake/java" in paths
        assert "/fake/dotnet" in paths
