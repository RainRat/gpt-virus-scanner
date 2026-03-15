from gptscan import Config

def test_is_supported_file_shebang_memory():
    # Test that in-memory content with shebang is correctly identified
    # even when it's just #!python without a space or /

    # Cases that used to fail because #! was not stripped and regex required / or space or start of string
    assert Config.is_supported_file("virtual.txt", content=b"#!python\n") is True
    assert Config.is_supported_file("virtual.txt", content=b"#!node\n") is True

    # Cases that always worked
    assert Config.is_supported_file("virtual.txt", content=b"#!/usr/bin/python\n") is True
    assert Config.is_supported_file("virtual.txt", content=b"#! /usr/bin/python\n") is True

    # Negative cases
    assert Config.is_supported_file("virtual.txt", content=b"#!notaninterpreter\n") is False
    assert Config.is_supported_file("virtual.txt", content=b"plain text\n") is False
