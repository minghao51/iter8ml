"""Versioned events and a durable per-run JSONL event sink."""

from __future__ import annotations

import gzip
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, field_validator

from iter8ml.domain.hashing import digest


class EventEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str
    attempt: int = Field(default=1, ge=1)
    stage: str | None = None
    node: str | None = None
    graph_version: str = Field(default_factory=lambda: digest("unknown"))
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("graph_version")
    @classmethod
    def validate_graph_version(cls, value: str) -> str:
        if not (value.startswith("sha256:") and len(value) == 71):
            raise ValueError("graph_version must be a full sha256 digest")
        if value[7:] != value[7:].lower():
            raise ValueError("graph_version must use lowercase hexadecimal characters")
        int(value[7:], 16)
        return value


class JsonlEventSink:
    """Append hot events and atomically publish a gzip archive on finalize."""

    def __init__(self, directory: str | Path, run_id: str, *, graph_version: str | None = None):
        self.directory = Path(directory)
        self.run_id = run_id
        self.graph_version = graph_version or digest("unknown")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hot_path = self.directory / f"{run_id}.jsonl"
        self.archive_path = self.directory / f"{run_id}.events.jsonl.gz"
        self.lock = FileLock(str(self.hot_path) + ".lock")
        self._finalized = self.archive_path.exists() and not self.hot_path.exists()

    def append(self, event: EventEnvelope) -> None:
        if event.run_id != self.run_id:
            raise ValueError("event run_id does not match sink run_id")
        if self._finalized:
            raise RuntimeError("event sink is already finalized")
        with self.lock, self.hot_path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def finalize(self, terminal_status: str) -> Path:
        if self._finalized:
            return self.archive_path
        fd, temp_name = tempfile.mkstemp(prefix=f"{self.run_id}.", suffix=".gz", dir=self.directory)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            with self.lock:
                if not self.hot_path.exists():
                    self.hot_path.touch()
                terminal_event = EventEnvelope(
                    event_type="run.finalized",
                    run_id=self.run_id,
                    graph_version=self.graph_version,
                    payload={"terminal_status": terminal_status},
                )
                with self.hot_path.open("rb") as source, gzip.open(temp_path, "wb") as target:
                    for line in source:
                        EventEnvelope.model_validate_json(line)
                        target.write(line)
                    target.write(terminal_event.model_dump_json().encode("utf-8") + b"\n")
                archive_fd = os.open(temp_path, os.O_RDONLY)
                try:
                    os.fsync(archive_fd)
                finally:
                    os.close(archive_fd)
                os.replace(temp_path, self.archive_path)
                self.hot_path.unlink()
                directory_fd = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                self._finalized = True
                return self.archive_path
        finally:
            temp_path.unlink(missing_ok=True)
