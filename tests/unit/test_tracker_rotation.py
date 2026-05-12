"""Tests for JSONLTracker log rotation."""

import json
import tempfile
from pathlib import Path

from iter8ml.engine.tracker import JSONLTracker


def test_log_rotation_when_file_exceeds_limit():
    """Test that log rotation occurs when file size exceeds limit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        tracker = JSONLTracker(
            log_path=str(log_path), max_file_size_mb=0.001, backup_count=3
        )  # 1KB limit

        # Write events until we exceed the limit
        for i in range(100):
            tracker.log_event({"event": "test", "iteration": i})

        # Check that rotation occurred
        assert log_path.exists()  # Current log
        backup_1 = log_path.with_suffix(".jsonl.1")
        assert backup_1.exists(), "First backup should exist after rotation"


def test_multiple_rotations_keep_specified_backups():
    """Test that only specified number of backups are kept."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        tracker = JSONLTracker(log_path=str(log_path), max_file_size_mb=0.001, backup_count=2)

        # Write enough events to cause multiple rotations
        for i in range(300):
            tracker.log_event({"event": "test", "iteration": i})

        # Check that we have the correct number of backups
        assert log_path.exists()  # Current log
        assert log_path.with_suffix(".jsonl.1").exists()  # First backup
        assert log_path.with_suffix(".jsonl.2").exists()  # Second backup
        assert not log_path.with_suffix(".jsonl.3").exists()  # Third backup should not exist


def test_thread_safe_writes():
    """Test that concurrent writes are thread-safe."""
    import threading

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        tracker = JSONLTracker(log_path=str(log_path), max_file_size_mb=10.0, backup_count=3)

        def write_events(thread_id: int):
            for i in range(50):
                tracker.log_event({"event": "test", "thread": thread_id, "iteration": i})

        # Launch multiple threads writing concurrently
        threads = [threading.Thread(target=write_events, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all events were written
        with open(log_path) as f:
            events = [json.loads(line) for line in f if line.strip()]
        assert len(events) == 250  # 5 threads * 50 events


def test_tracker_metadata_fields():
    """Test that tracker adds run_id and timestamp to events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        tracker = JSONLTracker(log_path=str(log_path))
        tracker.current_run_id = "test_run_123"

        tracker.log_event({"event": "custom_event", "data": "test"})

        with open(log_path) as f:
            event = json.loads(f.readline())

        assert event["run_id"] == "test_run_123"
        assert "timestamp" in event
        assert event["event"] == "custom_event"
        assert event["data"] == "test"
