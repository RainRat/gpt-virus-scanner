import io
import zipfile
import pytest
from gptscan import unpack_content

def test_unpack_content_recursion_limit_stops_at_depth_six():
    current_content = b"print('depth 6')"
    current_name = "leaf.py"

    for i in range(1, 8):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as z:
            z.writestr(current_name, current_content)
        current_content = buffer.getvalue()
        current_name = f"nest{i}.zip"

    results = list(unpack_content("outer.zip", current_content))
    assert len(results) == 0

def test_unpack_content_works_at_depth_five():
    current_content = b"print('at depth 5')"
    current_name = "leaf.py"

    for i in range(1, 5):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as z:
            z.writestr(current_name, current_content)
        current_content = buffer.getvalue()
        current_name = f"nest{i}.zip"

    results = list(unpack_content("nest4.zip", current_content))
    assert len(results) == 1
    assert "leaf.py" in results[0][0]
    assert results[0][1] == b"print('at depth 5')"
