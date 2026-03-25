import os
import pytest
from gptscan import Config

def test_save_extensions(tmp_path):
    # Change to temporary directory to avoid overwriting real extensions.txt
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        Config.extensions_set = {".test1", ".test2"}
        Config.save_extensions()

        assert os.path.exists("extensions.txt")
        with open("extensions.txt", "r") as f:
            lines = f.read().splitlines()

        assert sorted(lines) == [".test1", ".test2"]
    finally:
        os.chdir(original_cwd)

def test_set_extensions_normalization():
    Config.set_extensions(["py", ".js", "  BAT  "])
    assert ".py" in Config.extensions_set
    assert ".js" in Config.extensions_set
    assert ".bat" in Config.extensions_set
    assert len(Config.extensions_set) == 3
