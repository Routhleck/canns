"""Tests for Jupyter notebook integration utilities."""

import pytest

from canns.analyzer.visualization.core.jupyter_utils import is_jupyter_environment


def test_is_jupyter_environment_returns_false_in_pytest():
    """Test that is_jupyter_environment returns False when running in pytest."""
    # When running in pytest (not in a Jupyter notebook), this should return False
    result = is_jupyter_environment()
    assert result is False, "Expected False when not running in Jupyter notebook"


def test_is_jupyter_environment_returns_bool():
    """Test that is_jupyter_environment always returns a boolean."""
    result = is_jupyter_environment()
    assert isinstance(result, bool), "is_jupyter_environment should return a boolean"
