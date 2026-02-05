import importlib.metadata as metadata
import sys
import types


def test_cli_version_uses_metadata(monkeypatch, capsys):
    from canns.__main__ import main

    monkeypatch.setattr(metadata, "version", lambda _: "9.9.9")
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "9.9.9"


def test_cli_version_fallback(monkeypatch, capsys):
    from canns.__main__ import main
    import importlib

    def _raise(_: str) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(metadata, "version", _raise)
    version_mod = importlib.import_module("canns._version")
    monkeypatch.setattr(version_mod, "__version__", "0.0.0-test", raising=False)

    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "0.0.0-test"


def _install_pipeline_stubs(monkeypatch, calls):
    pipeline_mod = types.ModuleType("canns.pipeline")
    monkeypatch.setitem(sys.modules, "canns.pipeline", pipeline_mod)

    for name in ("asa", "gallery", "launcher", "asa_gui"):
        mod = types.ModuleType(f"canns.pipeline.{name}")

        def _make_main(key):
            def _main():
                calls[key] += 1

            return _main

        mod.main = _make_main(name)
        monkeypatch.setitem(sys.modules, f"canns.pipeline.{name}", mod)


def test_cli_dispatches_asa(monkeypatch):
    from canns.__main__ import main

    calls = {"asa": 0, "gallery": 0, "launcher": 0, "asa_gui": 0}
    _install_pipeline_stubs(monkeypatch, calls)

    assert main(["--asa"]) == 0
    assert calls["asa"] == 1


def test_cli_dispatches_gallery(monkeypatch):
    from canns.__main__ import main

    calls = {"asa": 0, "gallery": 0, "launcher": 0, "asa_gui": 0}
    _install_pipeline_stubs(monkeypatch, calls)

    assert main(["--gallery"]) == 0
    assert calls["gallery"] == 1


def test_cli_dispatches_gui(monkeypatch):
    from canns.__main__ import main

    calls = {"asa": 0, "gallery": 0, "launcher": 0, "asa_gui": 0}
    _install_pipeline_stubs(monkeypatch, calls)
    monkeypatch.setattr(sys.modules["canns.pipeline.asa_gui"], "main", lambda: 0)

    assert main(["--gui"]) == 0


def test_cli_dispatches_default(monkeypatch):
    from canns.__main__ import main

    calls = {"asa": 0, "gallery": 0, "launcher": 0, "asa_gui": 0}
    _install_pipeline_stubs(monkeypatch, calls)

    assert main([]) == 0
    assert calls["launcher"] == 1
