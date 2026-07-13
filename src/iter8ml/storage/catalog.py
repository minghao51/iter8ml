"""Rebuildable local catalog over committed manifests.

The file is named ``catalog.duckdb`` to preserve the planned public layout. The
implementation uses SQLite from the standard library so the core package stays
usable without adding a database dependency; callers can query the same tables.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from iter8ml.domain.manifests import ProductManifest, RunManifest
from iter8ml.storage.local import LocalArtifactStore


class LocalCatalogStore:
    def __init__(self, workspace_root: str | Path):
        self.path = Path(workspace_root) / "control" / "catalog" / "catalog.duckdb"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS products (
                    product_id TEXT PRIMARY KEY, product_type TEXT NOT NULL,
                    name TEXT NOT NULL, created_at TEXT NOT NULL,
                    manifest_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, run_key TEXT NOT NULL,
                    status TEXT NOT NULL, created_at TEXT NOT NULL,
                    manifest_json TEXT NOT NULL
                );
                """
            )

    def register_product(self, manifest: ProductManifest) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?)",
                (
                    manifest.product_id,
                    manifest.product_type,
                    manifest.name,
                    manifest.created_at.isoformat(),
                    manifest.model_dump_json(),
                ),
            )

    def register_run(self, manifest: RunManifest) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?)",
                (
                    manifest.run_id,
                    manifest.run_key,
                    manifest.status,
                    manifest.created_at.isoformat(),
                    manifest.model_dump_json(),
                ),
            )

    def rebuild(self, artifact_store: LocalArtifactStore) -> dict[str, int]:
        with self._connect() as conn:
            conn.execute("DELETE FROM products")
            conn.execute("DELETE FROM runs")
            products = list(artifact_store.list_products())
            for product in products:
                conn.execute(
                    "INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?)",
                    (
                        product.product_id,
                        product.product_type,
                        product.name,
                        product.created_at.isoformat(),
                        product.model_dump_json(),
                    ),
                )
            runs = []
            for path in (artifact_store.workspace_root / "control" / "runs").glob("*/run.json"):
                run = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
                runs.append(run)
                conn.execute(
                    "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?)",
                    (
                        run.run_id,
                        run.run_key,
                        run.status,
                        run.created_at.isoformat(),
                        run.model_dump_json(),
                    ),
                )
        return {"products": len(products), "runs": len(runs)}

    def query(self, sql: str, parameters: Sequence[Any] = ()) -> pl.DataFrame:
        with self._connect() as conn:
            return pl.read_database(
                query=sql, connection=conn, execute_options={"parameters": parameters}
            )

    def products(self) -> list[ProductManifest]:
        with self._connect() as conn:
            rows = conn.execute("SELECT manifest_json FROM products ORDER BY created_at").fetchall()
        return [ProductManifest.model_validate_json(row[0]) for row in rows]

    def export_summary(self, limit: int = 100) -> dict[str, Any]:
        with self._connect() as conn:
            products = conn.execute(
                "SELECT product_id, product_type, name, created_at "
                "FROM products ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            runs = conn.execute(
                "SELECT run_id, run_key, status, created_at "
                "FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return {
            "products": [
                dict(zip(("product_id", "product_type", "name", "created_at"), row, strict=True))
                for row in products
            ],
            "runs": [
                dict(zip(("run_id", "run_key", "status", "created_at"), row, strict=True))
                for row in runs
            ],
        }
