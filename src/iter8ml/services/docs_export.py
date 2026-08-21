"""Bounded, sanitized JSON projections for static documentation sites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from iter8ml.storage.catalog import LocalCatalogStore
from iter8ml.storage.local import LocalArtifactStore
from iter8ml.workspace import Workspace


class DocsExporter:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.store = LocalArtifactStore(workspace.root)
        self.catalog = LocalCatalogStore(workspace.root)

    def export(self, *, limit: int = 100) -> Path:
        output = self.workspace.site_data_dir
        output.mkdir(parents=True, exist_ok=True)
        products = list(self.store.list_products())

        # The catalog is the single read surface for runs; rebuild it from the
        # authoritative run.json files so it always mirrors disk.
        self.catalog.rebuild(self.store)
        run_manifests = list(reversed(self.catalog.runs()))[:limit]
        runs = [
            {
                "run_id": r.run_id,
                "run_key": r.run_key,
                "status": r.status,
                "created_at": r.created_at.isoformat(),
                "stages": [{"name": s.name, "status": s.status} for s in r.stages],
            }
            for r in run_manifests
        ]
        _write(output / "summary.json", {"products": len(products), "runs": len(runs)})
        _write(output / "runs" / "index.json", {"runs": runs})
        _write(
            output / "graphs" / "current.json",
            {
                "graph_version": "iter8ml.medallion.graph.v1",
                "edges": [
                    {"from": "bronze", "to": "silver"},
                    {"from": "silver", "to": "gold"},
                    {"from": "gold", "to": "platinum"},
                ],
            },
        )
        for r in run_manifests:
            _write(
                output / "runs" / f"{r.run_id}.json",
                {
                    "run_id": r.run_id,
                    "status": r.status,
                    "run_key": r.run_key,
                    "stages": [_project_stage(s.model_dump()) for s in r.stages],
                },
            )
        _write(
            output / "datasets" / "index.json",
            {
                "products": [
                    {
                        "product_id": product.product_id,
                        "product_type": product.product_type,
                        "name": product.name,
                        "created_at": product.created_at.isoformat(),
                        "inputs": product.inputs,
                        "quality_summary": product.quality_summary,
                    }
                    for product in products[-limit:]
                ]
            },
        )
        return output


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _project_stage(stage: dict[str, Any]) -> dict[str, Any]:
    """Allowlist operational fields; never export local paths or error payloads."""
    return {
        "name": stage.get("name"),
        "status": stage.get("status"),
        "attempt": stage.get("attempt"),
        "input_products": stage.get("input_products", []),
        "output_products": stage.get("output_products", []),
    }
