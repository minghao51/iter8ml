"""Versioned events and a durable per-run JSONL event sink."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field


class EventEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str
    attempt: int = 1
    stage: str | None = None
    node: str | None = None
    graph_version: str = "unknown"
    payload: dict[str, Any] = Field(default_factory=dict)


class JsonlEventSink:
    """Append hot events and atomically publish a gzip archive on finalize."""

    def __init__(self, directory: str | Path, run_id: str):
        self.directory = Path(directory)
        self.run_id = run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hot_path = self.directory / f"{run_id}.jsonl"
        self.archive_path = self.directory / f"{run_id}.events.jsonl.gz"
        self.lock = FileLock(str(self.hot_path) + ".lock")

    def append(self, event: EventEnvelope) -> None:
        if event.run_id != self.run_id:
            raise ValueError("event run_id does not match sink run_id")
        with self.lock, self.hot_path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def finalize(self, terminal_status: str) -> Path:
        if not self.hot_path.exists():
            self.hot_path.touch()
        fd, temp_name = tempfile.mkstemp(prefix=f"{self.run_id}.", suffix=".gz", dir=self.directory)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            with self.lock:
                with self.hot_path.open("rb") as source, gzip.open(temp_path, "wb") as target:
                    for line in source:
                        json.loads(line)
                        target.write(line)
                    target.write(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "event_type": "run.finalized",
                                "occurred_at": datetime.now(UTC).isoformat(),
                                "run_id": self.run_id,
                                "terminal_status": terminal_status,
                            }
                        ).encode("utf-8")
                        + b"\n"
                    )
                os.replace(temp_path, self.archive_path)
                return self.archive_path
        finally:
            temp_path.unlink(missing_ok=True)
