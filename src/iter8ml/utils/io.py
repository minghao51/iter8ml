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
import logging
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

_INTEGRITY_KEY = b"iter8ml_safe_dump_v1"

ErrorPolicy = Literal["raise", "skip", "skip_trailing"]


def load_events(path: str | Path, on_error: ErrorPolicy = "raise") -> list[dict[str, Any]]:
    """Load all events from a JSONL file (see iter_events for on_error)."""
    return list(iter_events(path, on_error=on_error))


def iter_events(path: str | Path, on_error: ErrorPolicy = "raise") -> Iterator[dict[str, Any]]:
    """Stream events from a JSONL file, one dict at a time.

    Malformed-line handling via on_error:
      - "raise": fail on the first malformed line (default; strict contract).
      - "skip": drop any malformed line, warning with the line number.
      - "skip_trailing": drop malformed lines only when they form a torn
        trailing tail (crash mid-write); a malformed line followed by a valid
        line is mid-file corruption and still raises.

    Note: skip/skip_trailing warnings (and the mid-file raise) materialize as
    the generator is consumed — drain it fully (see load_events) to surface
    them.
    """
    if on_error not in ("raise", "skip", "skip_trailing"):
        raise ValueError(f"Unknown on_error policy: {on_error!r}")
    path = Path(path)
    if not path.exists():
        return

    torn_tail: list[tuple[int, str]] = []
    valid_count = 0
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as e:
                if on_error == "raise":
                    raise ValueError(f"Invalid JSON at line {line_num} in {path}: {e}") from e
                if on_error == "skip":
                    logger.warning("Skipping malformed line %d in %s: %s", line_num, path, e)
                    continue
                torn_tail.append((line_num, str(e)))
                continue
            if torn_tail:
                # A valid line follows malformed ones: mid-file corruption,
                # not a torn tail.
                first_num, first_err = torn_tail[0]
                raise ValueError(f"Invalid JSON at line {first_num} in {path}: {first_err}")
            valid_count += 1
            yield event

    if on_error != "raise" and valid_count == 0 and len(torn_tail) > 1:
        logger.warning(
            "No valid JSON lines in %s: all %d line(s) malformed — data loss likely",
            path,
            len(torn_tail),
        )
    for line_num, err in torn_tail:
        logger.warning(
            "Skipping malformed trailing line %d in %s (torn final write?): %s",
            line_num,
            path,
            err,
        )


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
