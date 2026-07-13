"""Manifest and artifact verification entrypoints."""

from __future__ import annotations

from iter8ml.storage.local import LocalArtifactStore


def verify_product(
    store: LocalArtifactStore, product_id: str, *, deep: bool = True
) -> dict[str, object]:
    result = store.verify(product_id, deep=deep)
    if not result["ok"]:
        return result
    return {**result, "manifest_valid": True}
