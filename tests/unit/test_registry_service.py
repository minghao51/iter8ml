"""Test RegistryService."""

import json
from pathlib import Path

import pytest

from core.services.registry_service import RegistryService


@pytest.fixture
def temp_registry(tmp_path):
    """Create a temporary registry file."""
    return tmp_path / "registry.json"


def test_load_empty_registry(temp_registry):
    """Test loading non-existent registry returns empty dict."""
    service = RegistryService(temp_registry)
    assert service.load() == {}


def test_load_existing_registry(temp_registry):
    """Test loading existing registry."""
    temp_registry.write_text(json.dumps({"key1": {"model": "catboost"}}))
    service = RegistryService(temp_registry)
    assert service.load() == {"key1": {"model": "catboost"}}


def test_update_if_better_new_key(temp_registry):
    """Test updating registry with new key."""
    service = RegistryService(temp_registry)
    result = service.update_if_better("key1", "catboost", "run1", 0.95, "/path/to/model")
    assert result is True
    registry = service.load()
    assert registry["key1"]["score"] == 0.95


def test_update_if_better_higher_score(temp_registry):
    """Test updating registry with higher score."""
    temp_registry.write_text(json.dumps({"key1": {"score": 0.90}}))
    service = RegistryService(temp_registry)
    result = service.update_if_better("key1", "catboost", "run2", 0.95, "/path/to/model")
    assert result is True
    assert service.load()["key1"]["score"] == 0.95


def test_update_if_better_lower_score(temp_registry):
    """Test that lower score doesn't update registry."""
    temp_registry.write_text(json.dumps({"key1": {"score": 0.95}}))
    service = RegistryService(temp_registry)
    result = service.update_if_better("key1", "catboost", "run2", 0.90, "/path/to/model")
    assert result is False
    assert service.load()["key1"]["score"] == 0.95
