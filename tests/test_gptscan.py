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


def test_run_cli_reports_progress(monkeypatch, capsys):
    def fake_scan_files(_path, _deep, _show_all, _use_gpt, _cancel_event=None, **_kwargs):
        yield ('progress', (0, 2, None))
        yield ('result', ("/tmp/a", "10%", "", "", "", "snippet"))
        yield ('progress', (2, 2, None))

    monkeypatch.setattr(gptscan, "scan_files", fake_scan_files)

    gptscan.run_cli("/tmp", deep=False, show_all=False, use_gpt=False, rate_limit=1)

    captured = capsys.readouterr()
    # The progress messages now have leading/trailing \r and are padded to 80 chars
    assert "Scanning: 0/2 files" in captured.err
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
    monkeypatch.setattr(gptscan, "collect_files", lambda _targets: [sample_file])
    gptscan.Config.set_extensions([".txt"], missing=False)

    list(gptscan.scan_files(str(tmp_path), deep_scan=False, show_all=True, use_gpt=False, cancel_event=None))

    assert load_calls["count"] == 1
