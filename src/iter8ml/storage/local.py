"""Atomic local filesystem artifact store."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO

import polars as pl
from filelock import FileLock

from iter8ml.domain.manifests import ArtifactRef, ProductManifest, ProductType
from iter8ml.exceptions import ArtifactError

_LAYER_DIRS: dict[ProductType, str] = {
    "bronze": "01_bronze",
    "silver": "02_silver",
    "gold": "03_gold",
    "platinum": "04_platinum",
}
_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


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
            artifact_path = self.temp_dir / relative
            if not artifact_path.is_file():
                raise ArtifactError(f"Manifest references missing artifact: {relative}")
            if (
                artifact_path.stat().st_size != ref.size_bytes
                or _sha256(artifact_path) != ref.sha256
            ):
                raise ArtifactError(f"Manifest metadata does not match artifact: {relative}")
        manifest_path = self.temp_dir / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        for path in self.temp_dir.rglob("*"):
            if path.is_file():
                _fsync_file(path)
        (self.temp_dir / "_SUCCESS").touch()
        _fsync_file(self.temp_dir / "_SUCCESS")
        _fsync_directory(self.temp_dir)
        self.final_dir.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.store.locks_dir / f"{self.product_id}.lock")):
            if self.final_dir.exists():
                existing = self.store.read_manifest(self.product_id)
                if _equivalent_manifest(existing, manifest):
                    verification = self.store.verify(self.product_id, deep=True)
                    if not verification["ok"]:
                        raise ArtifactError(
                            f"Existing product failed verification: {self.product_id}"
                        )
                    self.abort()
                    return existing
                raise ArtifactError(
                    f"Committed product already exists with different content: {self.product_id}"
                )
            os.replace(self.temp_dir, self.final_dir)
            _fsync_directory(self.final_dir.parent)
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
        self.locks_dir = self.workspace_root / "control" / "locks"
        self.lake_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)

    def product_dir(self, product_type: ProductType, name: str, product_id: str) -> Path:
        _validate_component(name, "product name")
        _validate_component(product_id, "product_id")
        return self.lake_dir / _LAYER_DIRS[product_type] / name / product_id

    def begin(self, product_id: str, *, product_type: ProductType, name: str) -> ProductWriter:
        if self.exists(product_id):
            raise ArtifactError(f"Product already exists: {product_id}")
        return ProductWriter(self, product_id, product_type, name)

    def exists(self, product_id: str) -> bool:
        return any(path.parent.name == product_id for path in self.lake_dir.glob("*/**/_SUCCESS"))

    def read_manifest(self, product_id: str) -> ProductManifest:
        path = self._manifest_path(product_id, require_success=True)
        return ProductManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def read_verified_manifest(self, product_id: str, *, deep: bool = True) -> ProductManifest:
        verification = self.verify(product_id, deep=deep)
        if not verification["ok"]:
            raise ArtifactError(f"Product failed verification: {product_id}")
        return self.read_manifest(product_id)

    def open_artifact(self, ref: ArtifactRef) -> BinaryIO:
        product_id = ref.artifact_id.split(":", 1)[0]
        manifest = self.read_manifest(product_id)
        if ref not in manifest.artifacts:
            raise ArtifactError(f"Artifact is not declared by product manifest: {ref.uri}")
        path = self._path_for_uri(ref.uri)
        return path.open("rb")

    def _path_for_uri(self, uri: str) -> Path:
        try:
            product_id, relative = uri.split("/", 1)
        except ValueError as exc:
            raise ArtifactError(f"Invalid artifact URI: {uri}") from exc
        manifest_path = self._manifest_path(product_id, require_success=True)
        path = manifest_path.parent / relative
        if path.is_file() and path.resolve().is_relative_to(self.lake_dir.resolve()):
            return path
        raise ArtifactError(f"Artifact not found: {uri}")

    def _manifest_path(self, product_id: str, *, require_success: bool) -> Path:
        for path in self.lake_dir.glob("*/**/manifest.json"):
            if path.parent.name != product_id:
                continue
            if require_success and not (path.parent / "_SUCCESS").is_file():
                break
            return path
        state = "Committed product" if require_success else "Product"
        raise ArtifactError(f"{state} manifest not found: {product_id}")

    def verify(self, product_id: str, *, deep: bool = False) -> dict[str, object]:
        manifest_path = self._manifest_path(product_id, require_success=False)
        manifest = ProductManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        product_dir = manifest_path.parent
        success = (product_dir / "_SUCCESS").exists()
        checked = 0
        errors: list[str] = []
        if manifest.product_id != product_id:
            errors.append(
                f"manifest product_id mismatch: expected {product_id}, got {manifest.product_id}"
            )
        for ref in manifest.artifacts:
            try:
                relative = ref.uri.split("/", 1)[1]
                path = product_dir / relative
                if not path.is_file() or not path.resolve().is_relative_to(product_dir.resolve()):
                    raise ArtifactError(f"Artifact not found: {ref.uri}")
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
                if not (path.parent / "_SUCCESS").is_file():
                    continue
                yield ProductManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _equivalent_manifest(left: ProductManifest, right: ProductManifest) -> bool:
    return left.model_dump(exclude={"created_at"}) == right.model_dump(exclude={"created_at"})


def _validate_component(value: str, label: str) -> None:
    if not _PATH_COMPONENT.fullmatch(value):
        raise ArtifactError(f"Invalid {label}: {value!r}")


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _media_type(path: Path) -> str:
    return {
        ".json": "application/json",
        ".parquet": "application/vnd.apache.parquet",
        ".gz": "application/gzip",
    }.get(path.suffix, "application/octet-stream")
