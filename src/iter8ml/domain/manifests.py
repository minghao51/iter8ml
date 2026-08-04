"""Pydantic contracts for durable products and run control state."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from iter8ml.domain.hashing import digest

JsonValue = str | int | float | bool | None | dict[str, Any] | list[Any]
ProductType = Literal["bronze", "silver", "gold", "platinum"]
ArtifactKind = Literal[
    "dataset",
    "features",
    "labels",
    "splits",
    "transformer",
    "model",
    "predictions",
    "metrics",
    "quality",
    "report",
    "events",
]
StageName = Literal["bronze", "silver", "gold", "platinum", "publish"]
RunStatus = Literal["planned", "running", "succeeded", "partial", "failed", "cancelled"]


def _validate_digest(value: str) -> str:
    if not (value.startswith("sha256:") and len(value) == 71):
        raise ValueError("digest must be a full sha256:<64 lowercase hex> value")
    if value[7:] != value[7:].lower():
        raise ValueError("digest must use lowercase hexadecimal characters")
    int(value[7:], 16)
    return value


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceSpec(ContractModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    source_type: Literal["csv", "parquet", "sqlite", "memory"]
    uri: str
    query: str | None = None
    fingerprint: str | None = None

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str | None) -> str | None:
        return _validate_digest(value) if value is not None else None


class ArtifactRef(ContractModel):
    artifact_id: str
    kind: ArtifactKind
    uri: str
    media_type: str
    sha256: str
    size_bytes: int = Field(ge=0)
    row_count: int | None = Field(default=None, ge=0)
    schema_ref: str | None = None

    @field_validator("sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _validate_digest(value)


class LineageEdge(ContractModel):
    upstream: str
    downstream: str
    relation: Literal["derived_from", "consumes", "produces"] = "derived_from"


class ProductManifest(ContractModel):
    schema_version: Literal[1] = 1
    product_id: str
    product_type: ProductType
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: Literal["committed"] = "committed"
    inputs: list[str] = Field(default_factory=list)
    specification_digest: str
    code_digest: str
    graph_version: str
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    schema_digest: str | None = None
    quality_summary: dict[str, JsonValue] = Field(default_factory=dict)
    lineage: list[LineageEdge] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("specification_digest", "code_digest", "graph_version", "schema_digest")
    @classmethod
    def validate_digests(cls, value: str | None) -> str | None:
        return _validate_digest(value) if value is not None else None


class SplitSpec(ContractModel):
    strategy: Literal["kfold", "stratified", "group", "time", "purged_time"] = "kfold"
    folds: int = Field(default=5, ge=2)
    shuffle: bool = False
    random_seed: int | None = None
    group_column: str | None = None
    time_column: str | None = None
    gap: int = Field(default=0, ge=0)
    embargo: int = Field(default=0, ge=0)


class SplitManifest(ContractModel):
    schema_version: Literal[1] = 1
    split_id: str
    dataset_version: str
    row_id_column: str = "row_id"
    spec: SplitSpec
    artifact: ArtifactRef
    fold_counts: dict[str, dict[str, int]]
    overlap_checks_passed: bool
    temporal_checks_passed: bool | None

    @field_validator("split_id")
    @classmethod
    def validate_split_id(cls, value: str) -> str:
        return _validate_digest(value)


class RunPlan(ContractModel):
    schema_version: Literal[1] = 1
    plan_name: str
    materialization: Literal["quick", "reproducible", "production"] = "reproducible"
    source: SourceSpec
    contract: dict[str, JsonValue] = Field(default_factory=dict)
    target: dict[str, JsonValue] = Field(default_factory=dict)
    split: SplitSpec = Field(default_factory=SplitSpec)
    features: dict[str, JsonValue] = Field(default_factory=dict)
    models: list[str] | Literal["auto"] = "auto"
    evaluation: dict[str, JsonValue] = Field(default_factory=dict)
    resources: dict[str, JsonValue] = Field(default_factory=dict)
    retry: dict[str, JsonValue] = Field(default_factory=dict)
    failure: dict[str, JsonValue] = Field(default_factory=dict)
    promotion: dict[str, JsonValue] = Field(default_factory=dict)
    documentation: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def run_key(self) -> str:
        return digest(self.model_dump(mode="json"))


class StageRecord(ContractModel):
    name: StageName
    status: Literal["pending", "running", "succeeded", "failed", "skipped", "cached"] = "pending"
    attempt: int = Field(default=0, ge=0)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    input_products: list[str] = Field(default_factory=list)
    output_products: list[str] = Field(default_factory=list)
    error: dict[str, str] | None = None


class RunManifest(ContractModel):
    schema_version: Literal[1] = 1
    run_id: str
    run_key: str
    status: RunStatus = "planned"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    ended_at: datetime | None = None
    plan: RunPlan
    graph_version: str
    stages: list[StageRecord] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    quality_gate: dict[str, JsonValue] | None = None
    promotion: dict[str, JsonValue] | None = None
    error: dict[str, str] | None = None
    event_archive: str | None = None

    @field_validator("run_key", "graph_version")
    @classmethod
    def validate_digests(cls, value: str) -> str:
        return _validate_digest(value)
