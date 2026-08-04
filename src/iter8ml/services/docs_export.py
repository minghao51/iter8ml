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
        runs = []
        for path in sorted(self.workspace.runs_dir.glob("*/run.json"), reverse=True)[:limit]:
            data = json.loads(path.read_text(encoding="utf-8"))
            runs.append(
                {
                    "run_id": data.get("run_id"),
                    "run_key": data.get("run_key"),
                    "status": data.get("status"),
                    "created_at": data.get("created_at"),
                    "stages": [
                        {"name": stage.get("name"), "status": stage.get("status")}
                        for stage in data.get("stages", [])
                    ],
                }
            )
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
        for path in sorted(self.workspace.runs_dir.glob("*/run.json"), reverse=True)[:limit]:
            data = json.loads(path.read_text(encoding="utf-8"))
            _write(
                output / "runs" / f"{data['run_id']}.json",
                {
                    "run_id": data.get("run_id"),
                    "status": data.get("status"),
                    "run_key": data.get("run_key"),
                    "stages": [_project_stage(stage) for stage in data.get("stages", [])],
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
