from __future__ import annotations

import importlib
import importlib.util
import runpy
import sys
import types


def _import_with_fake_pyside(monkeypatch, module_name: str):
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name == "PySide6":
            return object()
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    for name in list(sys.modules):
        if name == "canns.pipeline" or name.startswith("canns.pipeline."):
            sys.modules.pop(name, None)

    return importlib.import_module(module_name)


def test_pipeline_init_exports(monkeypatch):
    pipeline = _import_with_fake_pyside(monkeypatch, "canns.pipeline")

    assert hasattr(pipeline, "ASAApp")
    assert hasattr(pipeline, "GalleryApp")
    assert hasattr(pipeline, "launcher_main")


def test_launcher_main_gallery(monkeypatch):
    launcher = _import_with_fake_pyside(monkeypatch, "canns.pipeline.launcher")

    called = {"gallery": 0}

    class GalleryApp:
        def run(self):
            called["gallery"] += 1

    monkeypatch.setitem(
        sys.modules,
        "canns.pipeline.gallery",
        types.SimpleNamespace(GalleryApp=GalleryApp),
    )
    monkeypatch.setattr(launcher.ModePicker, "run", lambda self: "gallery")

    launcher.main()
    assert called["gallery"] == 1


def test_launcher_main_asa(monkeypatch):
    launcher = _import_with_fake_pyside(monkeypatch, "canns.pipeline.launcher")

    called = {"asa": 0}

    class ASAApp:
        def run(self):
            called["asa"] += 1

    monkeypatch.setitem(
        sys.modules,
        "canns.pipeline.asa",
        types.SimpleNamespace(ASAApp=ASAApp),
    )
    monkeypatch.setattr(launcher.ModePicker, "run", lambda self: "asa")

    launcher.main()
    assert called["asa"] == 1


def test_launcher_main_quit(monkeypatch):
    launcher = _import_with_fake_pyside(monkeypatch, "canns.pipeline.launcher")

    monkeypatch.setattr(launcher.ModePicker, "run", lambda self: None)
    launcher.main()


def test_pipeline_main_calls_launcher(monkeypatch):
    launcher = _import_with_fake_pyside(monkeypatch, "canns.pipeline.launcher")
    called = {"main": 0}

    def _main():
        called["main"] += 1

    monkeypatch.setattr(launcher, "main", _main)
    runpy.run_module("canns.pipeline.__main__", run_name="__main__")
    assert called["main"] == 1
