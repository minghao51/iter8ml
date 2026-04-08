"""Test JSONL utilities."""

import pytest
from core.utils.jsonl import load_events


def test_load_events_handles_empty_file(tmp_path):
    """Test loading empty JSONL file."""
    jsonl_file = tmp_path / "empty.jsonl"
    jsonl_file.write_text("")
    assert load_events(jsonl_file) == []


def test_load_events_handles_malformed_json(tmp_path):
    """Test loading JSONL with malformed line."""
    jsonl_file = tmp_path / "malformed.jsonl"
    jsonl_file.write_text('{"valid": true}\n{invalid json}\n{"also": "valid"}')
    with pytest.raises(ValueError, match="Invalid JSON at line"):
        load_events(jsonl_file)


def test_load_events_handles_nonexistent_path():
    """Test loading non-existent file returns empty list."""
    assert load_events("/nonexistent/path.jsonl") == []


def test_load_events_filters_blank_lines(tmp_path):
    """Test that blank lines are filtered out."""
    jsonl_file = tmp_path / "blanks.jsonl"
    jsonl_file.write_text('{"a": 1}\n\n{"b": 2}\n   \n{"c": 3}')
    events = load_events(jsonl_file)
    assert len(events) == 3
