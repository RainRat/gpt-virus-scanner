from gptscan import unpack_content

def test_dockerfile_instruction_switch_bug():
    content = b"""
RUN echo 1 \\
CMD echo 2
"""
    results = list(unpack_content("Dockerfile", content))
    # results[0] should be Instruction 1, results[1] should be Instruction 2
    assert len(results) == 2
    assert results[0][1] == b"echo 1"
    assert results[1][1] == b"echo 2"

if __name__ == "__main__":
    test_dockerfile_instruction_switch_bug()
