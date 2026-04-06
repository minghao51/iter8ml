"""JSONL utilities."""

import json
from pathlib import Path


def load_events(path: str | Path) -> list[dict]:
    """Load events from a JSONL file."""
    events = []
    path = Path(path)
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
    return events
