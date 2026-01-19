"""Tests for visualization backend utilities."""

import platform
import pytest


def test_get_multiprocessing_context_returns_tuple():
    """Test that get_multiprocessing_context returns a tuple of (context, method_name)."""
    from canns.analyzer.visualization.core.backend import get_multiprocessing_context

    # Test default behavior (prefer_fork=False)
    result = get_multiprocessing_context(prefer_fork=False)

    # Should return a tuple
    assert isinstance(result, tuple), "get_multiprocessing_context should return a tuple"
    assert len(result) == 2, "Tuple should have exactly 2 elements"

    ctx, method_name = result

    # Context should be either a multiprocessing context or None
    if ctx is not None:
        # Verify it has the expected Pool method
        assert hasattr(ctx, "Pool"), "Context should have a Pool method"

    # Method name should be a string or None
    assert method_name is None or isinstance(method_name, str), \
        "Method name should be a string or None"

    # If context is not None, method name should also not be None
    if ctx is not None:
        assert method_name in ["spawn", "fork"], \
            f"Method name should be 'spawn' or 'fork', got {method_name}"


def test_get_multiprocessing_context_spawn():
    """Test that spawn method works on all platforms."""
    from canns.analyzer.visualization.core.backend import get_multiprocessing_context

    # Default should use spawn (prefer_fork=False)
    ctx, method_name = get_multiprocessing_context(prefer_fork=False)

    if ctx is not None:
        # Should return spawn on all platforms when prefer_fork=False
        assert method_name == "spawn", \
            f"Expected 'spawn' method, got {method_name}"


def test_get_multiprocessing_context_fork_on_linux():
    """Test that fork method is used on Linux when preferred (without JAX)."""
    from canns.analyzer.visualization.core.backend import get_multiprocessing_context
    import sys

    # Only test fork on Linux
    if platform.system() != "Linux":
        pytest.skip("Fork test only applicable on Linux")

    # Check if JAX is loaded
    has_jax = any(name.startswith("jax") for name in sys.modules)

    ctx, method_name = get_multiprocessing_context(prefer_fork=True)

    if ctx is not None:
        if has_jax:
            # With JAX, should fall back to spawn
            assert method_name == "spawn", \
                "Should use 'spawn' when JAX is detected"
        else:
            # Without JAX, should use fork
            assert method_name == "fork", \
                "Should use 'fork' on Linux when JAX is not present"


def test_get_multiprocessing_context_consistency():
    """Test that the same parameters return consistent results."""
    from canns.analyzer.visualization.core.backend import get_multiprocessing_context

    # Call twice with same parameters
    ctx1, method1 = get_multiprocessing_context(prefer_fork=False)
    ctx2, method2 = get_multiprocessing_context(prefer_fork=False)

    # Method names should be consistent
    assert method1 == method2, \
        "Method name should be consistent across calls"

    # Both should be None or both should be not None
    assert (ctx1 is None) == (ctx2 is None), \
        "Context availability should be consistent"
