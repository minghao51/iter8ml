"""Tests for restricted pickle deserialization."""

import pickle

import pytest

from iter8ml.utils.io import safe_load


def test_safe_load_allows_whitelisted_builtin_containers():
    payload = pickle.dumps({"a": [1, 2, 3], "b": ("x", "y")}, protocol=pickle.HIGHEST_PROTOCOL)
    value = safe_load(payload)
    assert value == {"a": [1, 2, 3], "b": ("x", "y")}


def test_safe_load_blocks_non_whitelisted_builtin_globals():
    payload = pickle.dumps(eval, protocol=pickle.HIGHEST_PROTOCOL)
    with pytest.raises(pickle.UnpicklingError, match="Blocked deserialization"):
        safe_load(payload)
