"""Small helpers for human-readable identifiers."""

from __future__ import annotations

from datetime import UTC, datetime

from iter8ml.domain.hashing import digest


def product_id(product_type: str, name: str, *inputs: str) -> str:
    """Build a stable product identity from type, name, and upstream identities."""
    return f"{product_type}_{name}_{digest([product_type, name, *inputs])[7:19]}"


def run_id(run_key: str, now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{timestamp}_{run_key[7:19]}"
