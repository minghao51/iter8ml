"""Tests for TrackingHook: Hamilton NodeExecutionHook -> Tracker protocol."""

import json
from types import SimpleNamespace

from tabular_blueprint.engine.tracker import JSONLTracker
from tabular_blueprint.pipelines.hooks.tracking_hook import TrackingHook


def _read_events(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class TestTrackingHook:
    def test_node_error_logs_event(self, tmp_path):
        log_path = tmp_path / "experiments.jsonl"
        tracker = JSONLTracker(str(log_path))
        hook = TrackingHook(tracker, run_id="test_run")

        node = SimpleNamespace(name="some_node")
        hook.run_on_node_error(node, ValueError("boom"), None, None)

        events = list(_read_events(log_path))
        assert len(events) == 1
        assert events[0]["event"] == "node_error"
        assert events[0]["node"] == "some_node"
        assert events[0]["error"] == "boom"

    def test_node_success_logs_event(self, tmp_path):
        log_path = tmp_path / "experiments.jsonl"
        tracker = JSONLTracker(str(log_path))
        hook = TrackingHook(tracker, run_id="test_run")

        node = SimpleNamespace(name="fill_nulls_numeric")
        ctx = SimpleNamespace(duration=1.23456)
        hook.run_on_node_success(node, None, None, ctx)

        events = list(_read_events(log_path))
        assert len(events) == 1
        assert events[0]["event"] == "node_completed"
        assert events[0]["node"] == "fill_nulls_numeric"
        assert events[0]["duration_seconds"] == 1.2346

    def test_node_success_without_duration(self, tmp_path):
        log_path = tmp_path / "experiments.jsonl"
        tracker = JSONLTracker(str(log_path))
        hook = TrackingHook(tracker, run_id="test_run")

        node = SimpleNamespace(name="some_node")
        hook.run_on_node_success(node, None, None, SimpleNamespace(duration=None))

        events = list(_read_events(log_path))
        assert events[0]["event"] == "node_completed"
        assert "duration_seconds" not in events[0]

    def test_node_name_fallback_when_no_name_attr(self, tmp_path):
        log_path = tmp_path / "experiments.jsonl"
        tracker = JSONLTracker(str(log_path))
        hook = TrackingHook(tracker, run_id="test_run")

        hook.run_on_node_error("a_string_node", ValueError("err"), None, None)

        events = list(_read_events(log_path))
        assert events[0]["node"] == "a_string_node"

    def test_run_id_stored_in_tracker(self):
        tracker = JSONLTracker("/tmp/test.jsonl")
        hook = TrackingHook(tracker, run_id="exp_001")
        assert hook._run_id == "exp_001"

    def test_before_and_after_execution_are_noops(self, tmp_path):
        log_path = tmp_path / "experiments.jsonl"
        tracker = JSONLTracker(str(log_path))
        hook = TrackingHook(tracker)

        hook.run_before_node_execution(None, None, None)
        hook.run_after_node_execution(None, None, None, None)

        assert not log_path.exists() or len(list(_read_events(log_path))) == 0

    def test_node_error_with_no_run_id(self, tmp_path):
        log_path = tmp_path / "experiments.jsonl"
        tracker = JSONLTracker(str(log_path))
        hook = TrackingHook(tracker)

        node = SimpleNamespace(name="test_node")
        hook.run_on_node_error(node, RuntimeError("fail"), None, None)

        events = list(_read_events(log_path))
        assert events[0]["event"] == "node_error"
        assert events[0]["node"] == "test_node"
