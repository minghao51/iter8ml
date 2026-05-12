"""File and serialization utilities: JSONL + safe pickle."""

import io
import json
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def load_events(path: str | Path) -> list[dict[str, Any]]:
    """Load events from a JSONL file."""
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


def iter_events(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream events from a JSONL file, one dict at a time."""
    path = Path(path)
    if not path.exists():
        return

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if stripped:
                try:
                    yield json.loads(stripped)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num} in {path}: {e}") from e


WHITELISTED_PREFIXES: tuple[str, ...] = (
    "sklearn.",
    "numpy.",
    "scipy.",
    "catboost.",
    "lightgbm.",
    "xgboost.",
    "tabpfn.",
    "collections.",
)

WHITELISTED_CLASSES: frozenset[str] = frozenset(
    {
        "builtins.dict",
        "builtins.list",
        "builtins.tuple",
        "builtins.str",
        "builtins.int",
        "builtins.float",
        "builtins.NoneType",
        "builtins.bool",
        "builtins.set",
        "builtins.frozenset",
        "builtins.bytes",
        "collections.OrderedDict",
    }
)


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        fqn = f"{module}.{name}"
        if fqn in WHITELISTED_CLASSES:
            return super().find_class(module, name)
        for prefix in WHITELISTED_PREFIXES:
            if fqn.startswith(prefix):
                return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Blocked deserialization of '{fqn}'.")


def safe_load(data: bytes | io.BufferedIOBase) -> Any:
    if isinstance(data, (bytes, bytearray)):
        return RestrictedUnpickler(io.BytesIO(data)).load()
    return RestrictedUnpickler(data).load()


def safe_loads(data: str) -> Any:
    return RestrictedUnpickler(io.BytesIO(data.encode("latin-1"))).load()


def safe_load_file(path: str) -> Any:
    with open(path, "rb") as f:
        return safe_load(f)


def safe_dump(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
