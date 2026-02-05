import sys


def _clear_canns_modules():
    for name in list(sys.modules):
        if name == "canns" or name.startswith("canns."):
            sys.modules.pop(name, None)


def test_lazy_imports():
    _clear_canns_modules()

    import canns

    assert "canns.models" not in sys.modules
    _ = canns.models
    assert "canns.models" in sys.modules
