"""File and serialization utilities: JSONL + safe pickle.

Pickle integrity uses HMAC-SHA256 to detect accidental file corruption
(truncated writes, bit rot, partial transfers).  The key is embedded in
source and is NOT a security measure — an attacker with source access
can forge valid signatures.  For tamper-resistant storage, use
encryption at the application layer.
"""

import hashlib
import hmac
import io
import json
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_INTEGRITY_KEY = b"iter8ml_safe_dump_v1"


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


def _strip_hmac_header(data: bytes) -> bytes:
    newline = data.find(b"\n")
    if newline == -1:
        return data
    header = data[: newline + 1]
    payload = data[newline + 1 :]
    if header.startswith(b"HMAC-SHA256:"):
        expected = header[len("HMAC-SHA256:") : -1]
        computed = hmac.new(_INTEGRITY_KEY, payload, hashlib.sha256).hexdigest().encode()
        if not hmac.compare_digest(expected, computed):
            raise ValueError("Integrity check failed: data may have been tampered with")
        return payload
    return data


def safe_load(data: bytes | io.BufferedIOBase) -> Any:
    if isinstance(data, (bytes, bytearray)):
        stripped = _strip_hmac_header(data)
        return RestrictedUnpickler(io.BytesIO(stripped)).load()
    return RestrictedUnpickler(data).load()


def safe_loads(data: str) -> Any:
    return RestrictedUnpickler(io.BytesIO(data.encode("latin-1"))).load()


def safe_load_file(path: str) -> Any:
    with open(path, "rb") as f:
        raw = f.read()
    data = _strip_hmac_header(raw)
    return RestrictedUnpickler(io.BytesIO(data)).load()


def safe_dump(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    pickle.dump(obj, buf, protocol=pickle.HIGHEST_PROTOCOL)
    data = buf.getvalue()
    mac = hmac.new(_INTEGRITY_KEY, data, hashlib.sha256).hexdigest()
    with open(path, "wb") as f:
        f.write(f"HMAC-SHA256:{mac}\n".encode())
        f.write(data)
