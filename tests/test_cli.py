import importlib.metadata as metadata


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
