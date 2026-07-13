"""Explicit reachability-based cleanup for local products."""

from __future__ import annotations

import shutil

from iter8ml.storage.local import LocalArtifactStore


def garbage_collect(store: LocalArtifactStore, *, dry_run: bool = True) -> list[str]:
    """Remove only abandoned temporary product directories by default."""
    candidates = [path for path in store.lake_dir.glob("**/.*") if path.is_dir()]
    paths = [str(path) for path in candidates]
    if not dry_run:
        for path in candidates:
            shutil.rmtree(path, ignore_errors=True)
    return paths
