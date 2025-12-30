import asyncio
import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import gptscan


class _FakeTree:
    def __init__(self, rows, column_map=None):
        self.data = rows
        self.order = list(rows.keys())
        self.heading_calls = {}
        self.column_map = column_map or {}

    def set(self, iid, col):
        return self.data[iid]["values"][self.column_map.get(col, 0)]

    def get_children(self, *_):
        return tuple(self.order)

    def move(self, iid, _, index):
        self.order.remove(iid)
        self.order.insert(index, iid)

    def heading(self, col, command=None):
        self.heading_calls[col] = command

    def column(self, cid):
        return {"width": 10}

    def item(self, iid):
        return self.data[iid]


def test_load_file_single_line(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("first line\nsecond line")

    result = gptscan.load_file(str(file_path))

    assert result == "first line"


def test_load_file_multi_line(tmp_path):
    file_path = tmp_path / "lines.txt"
    file_path.write_text("line1\nline2\n")

    result = gptscan.load_file(str(file_path), mode="multi_line")

    assert result == ["line1", "line2"]


def test_load_file_missing_returns_empty_string():
    assert gptscan.load_file("non_existent_file.txt") == ""


def test_load_file_missing_multiline_returns_empty_list():
    assert gptscan.load_file("non_existent_file_multiline.txt", mode="multi_line") == []


class _MockChoice:
    def __init__(self, content: str):
        self.message = SimpleNamespace(content=content)


class _MockResponse:
    def __init__(self, content: str):
        self.choices = [_MockChoice(content)]


def test_extract_data_from_gpt_response_parses_json():
    response = _MockResponse(
        json.dumps(
            {
                "administrator": "Admin",
                "end-user": "User",
                "threat-level": 75,
            }
        )
    )

    parsed = gptscan.extract_data_from_gpt_response(response)

    assert parsed["administrator"] == "Admin"
    assert parsed["end-user"] == "User"
    assert parsed["threat-level"] == 75


def test_extract_data_from_gpt_response_invalid_missing_keys():
    response = _MockResponse(json.dumps({"administrator": "Only admin"}))

    error = gptscan.extract_data_from_gpt_response(response)

    assert "Missing keys" in error


def test_extract_data_from_gpt_response_invalid_threat_level():
    response = _MockResponse(
        json.dumps(
            {
                "administrator": "Admin",
                "end-user": "User",
                "threat-level": "high",  # not an integer
            }
        )
    )

    error = gptscan.extract_data_from_gpt_response(response)

    assert "not a valid integer" in error


def test_extract_data_from_gpt_response_rejects_extra_keys():
    response = _MockResponse(
        json.dumps(
            {
                "administrator": "Admin",
                "end-user": "User",
                "threat-level": 50,
                "extra": "field",
            }
        )
    )

    error = gptscan.extract_data_from_gpt_response(response)

    assert "Unexpected keys" in error


def test_extract_data_from_gpt_response_rejects_non_object():
    response = _MockResponse(json.dumps([1, 2, 3]))

    error = gptscan.extract_data_from_gpt_response(response)

    assert "must be an object" in error


def test_extract_data_from_gpt_response_none_root():
    response = _MockResponse("null")

    error = gptscan.extract_data_from_gpt_response(response)

    assert "must be an object" in error


def test_extract_data_from_gpt_response_coerces_threat_level_string():
    response = _MockResponse(
        json.dumps(
            {
                "administrator": "Admin",
                "end-user": "User",
                "threat-level": "80",
            }
        )
    )

    parsed = gptscan.extract_data_from_gpt_response(response)

    assert parsed["threat-level"] == 80


def test_extract_data_from_gpt_response_invalid_structure():
    with pytest.raises(AttributeError):
        gptscan.extract_data_from_gpt_response(object())


def test_sort_column_orders_treeview():
    tree = _FakeTree(
        {
            "item1": {"values": ("b", "10%")},
            "item2": {"values": ("a", "30%")},
        },
        column_map={"name": 0, "own_conf": 1},
    )

    gptscan.sort_column(tree, "name", False)
    assert tree.get_children("") == ("item2", "item1")

    gptscan.sort_column(tree, "own_conf", False)
    assert tree.get_children("") == ("item1", "item2")


def test_sort_column_empty_tree():
    tree = _FakeTree({}, column_map={"name": 0})

    gptscan.sort_column(tree, "name", False)

    assert tree.get_children("") == ()


def test_sort_column_single_item():
    tree = _FakeTree({"item1": {"values": ("only",)}}, column_map={"name": 0})

    gptscan.sort_column(tree, "name", False)

    assert tree.get_children("") == ("item1",)


def test_sort_column_equal_values():
    tree = _FakeTree(
        {
            "item1": {"values": ("same",)},
            "item2": {"values": ("same",)},
        },
        column_map={"name": 0},
    )

    gptscan.sort_column(tree, "name", False)

    assert tree.get_children("") == ("item1", "item2")


def test_sort_column_reverse_toggle():
    tree = _FakeTree(
        {
            "item1": {"values": ("b",)},
            "item2": {"values": ("a",)},
        },
        column_map={"name": 0},
    )

    gptscan.sort_column(tree, "name", False)
    assert tree.get_children("") == ("item2", "item1")

    # Trigger reverse sort using stored command
    tree.heading_calls["name"]()
    assert tree.get_children("") == ("item1", "item2")


def test_sort_column_with_invalid_percentages():
    tree = _FakeTree(
        {
            "item1": {"values": ("JSON Parse Error",)},
            "item2": {"values": ("",)},
            "item3": {"values": ("30%",)},
            "item4": {"values": ("5%",)},
        },
        column_map={"gpt_conf": 0},
    )

    gptscan.sort_column(tree, "gpt_conf", False)

    assert tree.get_children("") == ("item1", "item2", "item4", "item3")


def test_run_cli_reports_progress(monkeypatch, capsys):
    def fake_scan_files(_path, _deep, _show_all, _use_gpt, _cancel_event=None, **_kwargs):
        yield ('progress', (0, 2, None))
        yield ('result', ("/tmp/a", "10%", "", "", "", "snippet"))
        yield ('progress', (2, 2, None))

    monkeypatch.setattr(gptscan, "scan_files", fake_scan_files)

    gptscan.run_cli("/tmp", deep=False, show_all=False, use_gpt=False, rate_limit=1)

    captured = capsys.readouterr()
    assert "Scanning: 0/2 files\r" in captured.err
    assert "Scanning: 2/2 files" in captured.err
    assert "path,own_conf" not in captured.err


def test_adjust_newlines_wraps_text():
    text = "lorem ipsum dolor"

    wrapped = gptscan.adjust_newlines(text, width=6, pad=0, measure=len)

    assert wrapped.split("\n") == ["lorem", "ipsum", "dolor"]


def test_scan_files_uses_cached_model(monkeypatch, tmp_path):
    class FakeTensor:
        def __init__(self, data):
            self.data = data

        def __getitem__(self, key):
            if isinstance(key, tuple):
                slice_obj = key[1]
                return FakeTensor(self.data[slice_obj])
            return self.data[key]

    load_calls = {"count": 0}

    class FakeModel:
        def predict(self, *_args, **_kwargs):
            return [[0.1]]

    def fake_load_model(_path, compile=False):
        load_calls["count"] += 1
        return FakeModel()

    fake_models = SimpleNamespace(load_model=fake_load_model)
    fake_tf = SimpleNamespace(
        keras=SimpleNamespace(models=fake_models),
        constant=lambda data: FakeTensor(list(data)),
        expand_dims=lambda tensor, axis=0: tensor,
    )

    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("abc")

    monkeypatch.setattr(gptscan, "_tf_module", fake_tf, raising=False)
    monkeypatch.setattr(gptscan, "_model_cache", None, raising=False)
    monkeypatch.setattr(gptscan, "collect_files", lambda _targets, exclude_patterns=None: [sample_file])
    gptscan.Config.set_extensions([".txt"], missing=False)

    list(gptscan.scan_files(str(tmp_path), deep_scan=False, show_all=True, use_gpt=False, cancel_event=None))
    list(gptscan.scan_files(str(tmp_path), deep_scan=False, show_all=True, use_gpt=False, cancel_event=None))

    assert load_calls["count"] == 1


def test_scan_files_handles_permission_error(monkeypatch, tmp_path):
    blocked_file = tmp_path / "blocked.txt"
    blocked_file.write_text("content")

    gptscan.Config.set_extensions([".txt"], missing=False)
    monkeypatch.setattr(gptscan, "collect_files", lambda _targets, exclude_patterns=None: [blocked_file])

    # Mock TensorFlow model dependencies
    monkeypatch.setattr(gptscan, "get_model", lambda: SimpleNamespace(predict=lambda *a, **k: [[0.0]]))
    monkeypatch.setattr(gptscan, "_tf_module", SimpleNamespace(
        constant=lambda x: x,
        expand_dims=lambda x, axis: x
    ), raising=False)

    real_open = builtins.open

    def fake_open(path, mode='r', *args, **kwargs):
        if str(path) == str(blocked_file) and 'b' in mode:
            raise PermissionError("permission denied")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(gptscan, "open", fake_open, raising=False)

    results = list(gptscan.scan_files(str(tmp_path), deep_scan=False, show_all=True, use_gpt=False, cancel_event=None))

    error_result = next((event for event in results if event[0] == 'result'), None)
    assert error_result is not None
    assert error_result[1][1] == 'Error'
    assert "permission denied" in error_result[1][5]


def test_async_handle_gpt_response_uses_cache(monkeypatch):
    gptscan.Config.gpt_cache = {}

    completions = SimpleNamespace(calls=[])

    async def fake_create(messages=None, model=None):
        completions.calls.append(messages)
        return _MockResponse(
            json.dumps(
                {
                    "administrator": "Admin",
                    "end-user": "User",
                    "threat-level": 42,
                }
            )
        )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))

    monkeypatch.setattr(gptscan, "get_async_openai_client", lambda: fake_client)

    limiter = gptscan.AsyncRateLimiter(10)
    semaphore = asyncio.Semaphore(2)

    first = asyncio.run(
        gptscan.async_handle_gpt_response(
            "code snippet",
            "task",
            rate_limiter=limiter,
            semaphore=semaphore,
        )
    )
    second = asyncio.run(
        gptscan.async_handle_gpt_response(
            "code snippet",
            "task",
            rate_limiter=limiter,
            semaphore=semaphore,
        )
    )

    assert first == second
    assert len(completions.calls) == 1
