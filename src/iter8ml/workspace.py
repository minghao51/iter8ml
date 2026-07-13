from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_ENV_WORKSPACE = "ITER8ML_WORKSPACE"
_DEFAULT_ROOT = "workspace"

__all__ = ["Workspace"]


def _default_root() -> Path:
    return Path(os.environ.get(_ENV_WORKSPACE, _DEFAULT_ROOT))


@dataclass
class Workspace:
    root: Path = field(default_factory=_default_root)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def experiments_path(self) -> Path:
        return self.root / "experiments.jsonl"

    @property
    def registry_path(self) -> Path:
        return self.root / "registry.json"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def lake_dir(self) -> Path:
        return self.root / "lake"

    @property
    def control_dir(self) -> Path:
        return self.root / "control"

    @property
    def runs_dir(self) -> Path:
        return self.control_dir / "runs"

    @property
    def events_dir(self) -> Path:
        return self.control_dir / "events"

    @property
    def catalog_path(self) -> Path:
        return self.control_dir / "catalog" / "catalog.duckdb"

    @property
    def site_data_dir(self) -> Path:
        return self.root / "site-data"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def state_path(self) -> Path:
        return self.root / "current_state.md"

    @property
    def leaderboard_path(self) -> Path:
        return self.root / "leaderboard.md"

    def init(self) -> Workspace:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.lake_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        (self.control_dir / "catalog").mkdir(parents=True, exist_ok=True)
        self.site_data_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_path.touch(exist_ok=True)
        if not self.registry_path.exists():
            self.registry_path.write_text("{}")
        return self
