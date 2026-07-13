"""Atomic local filesystem artifact store."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO

import polars as pl

from iter8ml.domain.manifests import ArtifactRef, ProductManifest, ProductType
from iter8ml.exceptions import ArtifactError

_LAYER_DIRS: dict[ProductType, str] = {
    "bronze": "01_bronze",
    "silver": "02_silver",
    "gold": "03_gold",
    "platinum": "04_platinum",
}


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


class ProductWriter:
    def __init__(
        self, store: LocalArtifactStore, product_id: str, product_type: ProductType, name: str
    ):
        self.store = store
        self.product_id = product_id
        self.product_type = product_type
        self.name = name
        self.final_dir = store.product_dir(product_type, name, product_id)
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f".{product_id}.", dir=store.lake_dir))
        self._closed = False

    def _target(self, relative_path: str) -> Path:
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ArtifactError(f"Artifact path escapes product directory: {relative_path}")
        target = self.temp_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def add_file(
        self,
        source: str | Path,
        *,
        relative_path: str,
        kind: str,
        row_count: int | None = None,
    ) -> ArtifactRef:
        source_path = Path(source)
        if not source_path.is_file():
            raise ArtifactError(f"Artifact source does not exist: {source_path}")
        target = self._target(relative_path)
        shutil.copy2(source_path, target)
        return ArtifactRef(
            artifact_id=f"{self.product_id}:{relative_path}",
            kind=kind,  # type: ignore[arg-type]
            uri=f"{self.product_id}/{relative_path}",
            media_type=_media_type(target),
            sha256=_sha256(target),
            size_bytes=target.stat().st_size,
            row_count=row_count,
        )

    def write_json(self, value: object, *, relative_path: str, kind: str) -> ArtifactRef:
        target = self._target(relative_path)
        target.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        return self._ref(target, relative_path, kind)

    def write_parquet(self, frame: pl.DataFrame, *, relative_path: str, kind: str) -> ArtifactRef:
        target = self._target(relative_path)
        frame.write_parquet(target)
        return self._ref(target, relative_path, kind, row_count=frame.height)

    def _ref(
        self, target: Path, relative_path: str, kind: str, row_count: int | None = None
    ) -> ArtifactRef:
        return ArtifactRef(
            artifact_id=f"{self.product_id}:{relative_path}",
            kind=kind,  # type: ignore[arg-type]
            uri=f"{self.product_id}/{relative_path}",
            media_type=_media_type(target),
            sha256=_sha256(target),
            size_bytes=target.stat().st_size,
            row_count=row_count,
        )

    def commit(self, manifest: ProductManifest) -> ProductManifest:
        if self._closed:
            raise ArtifactError("Product writer is already closed")
        if manifest.product_id != self.product_id:
            raise ArtifactError("Manifest product_id does not match writer")
        if manifest.product_type != self.product_type or manifest.name != self.name:
            raise ArtifactError("Manifest product type/name does not match writer")
        for ref in manifest.artifacts:
            relative = ref.uri.split("/", 1)[1] if "/" in ref.uri else ref.uri
            if not (self.temp_dir / relative).is_file():
                raise ArtifactError(f"Manifest references missing artifact: {relative}")
        manifest_path = self.temp_dir / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (self.temp_dir / "_SUCCESS").touch()
        self.final_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.final_dir.exists():
            existing = self.store.read_manifest(self.product_id)
            if existing == manifest:
                self.abort()
                return existing
            raise ArtifactError(
                f"Committed product already exists with different content: {self.product_id}"
            )
        os.replace(self.temp_dir, self.final_dir)
        self._closed = True
        return manifest

    def abort(self) -> None:
        if not self._closed:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self._closed = True


class LocalArtifactStore:
    """Content-addressed enough local store with atomic product directories."""

    def __init__(self, workspace_root: str | Path):
        self.workspace_root = Path(workspace_root)
        self.lake_dir = self.workspace_root / "lake"
        self.lake_dir.mkdir(parents=True, exist_ok=True)

    def product_dir(self, product_type: ProductType, name: str, product_id: str) -> Path:
        return self.lake_dir / _LAYER_DIRS[product_type] / name / product_id

    def begin(self, product_id: str, *, product_type: ProductType, name: str) -> ProductWriter:
        if self.exists(product_id):
            raise ArtifactError(f"Product already exists: {product_id}")
        return ProductWriter(self, product_id, product_type, name)

    def exists(self, product_id: str) -> bool:
        return any(
            path.parent.name == product_id for path in self.lake_dir.glob("*/**/manifest.json")
        )

    def read_manifest(self, product_id: str) -> ProductManifest:
        for path in self.lake_dir.glob("*/**/manifest.json"):
            if path.parent.name == product_id:
                return ProductManifest.model_validate_json(path.read_text(encoding="utf-8"))
        raise ArtifactError(f"Product manifest not found: {product_id}")

    def open_artifact(self, ref: ArtifactRef) -> BinaryIO:
        product_id = ref.artifact_id.split(":", 1)[0]
        manifest = self.read_manifest(product_id)
        del manifest
        path = self._path_for_uri(ref.uri)
        return path.open("rb")

    def _path_for_uri(self, uri: str) -> Path:
        product_id, relative = uri.split("/", 1)
        for manifest_path in self.lake_dir.glob("*/**/manifest.json"):
            if manifest_path.parent.name == product_id:
                path = manifest_path.parent / relative
                if path.is_file() and path.resolve().is_relative_to(self.lake_dir.resolve()):
                    return path
        raise ArtifactError(f"Artifact not found: {uri}")

    def verify(self, product_id: str, *, deep: bool = False) -> dict[str, object]:
        manifest = self.read_manifest(product_id)
        product_dir = self._path_for_uri(f"{product_id}/manifest.json").parent
        success = (product_dir / "_SUCCESS").exists()
        checked = 0
        errors: list[str] = []
        for ref in manifest.artifacts:
            try:
                path = self._path_for_uri(ref.uri)
                checked += 1
                if path.stat().st_size != ref.size_bytes:
                    errors.append(f"size mismatch: {ref.uri}")
                if deep and _sha256(path) != ref.sha256:
                    errors.append(f"checksum mismatch: {ref.uri}")
            except ArtifactError as exc:
                errors.append(str(exc))
        return {
            "product_id": product_id,
            "ok": success and not errors,
            "success": success,
            "checked": checked,
            "errors": errors,
        }

    def list_products(self, product_type: ProductType | None = None) -> Iterable[ProductManifest]:
        layers = [_LAYER_DIRS[product_type]] if product_type else list(_LAYER_DIRS.values())
        for layer in layers:
            for path in (self.lake_dir / layer).glob("*/**/manifest.json"):
                yield ProductManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _media_type(path: Path) -> str:
    return {
        ".json": "application/json",
        ".parquet": "application/vnd.apache.parquet",
        ".gz": "application/gzip",
    }.get(path.suffix, "application/octet-stream")
