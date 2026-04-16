"""JSONL utilities."""

import json
from pathlib import Path


def load_events(path: str | Path) -> list[dict]:
    """Load events from a JSONL file.

    Args:
        path: Path to JSONL file.

    Returns:
        List of event dictionaries. Returns empty list if file doesn't exist.

    Raises:
        ValueError: If a line contains malformed JSON.
    """
    events = []
    path = Path(path)
    if not path.exists():
        return []

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if stripped:
                try:
                    events.append(json.loads(stripped))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num} in {path}: {e}") from e
    return events
