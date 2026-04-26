# Architecture

## Overview

Tabular Blueprint is a single-node tabular ML framework with a CLI-first workflow.
Core code lives under `src/tabular_blueprint` and follows a thin orchestration model:
- `cli.py` handles user entrypoints.
- `engine/` coordinates run orchestration, evaluation, HPO, tracking, and state generation.
- `data/` loads and prepares datasets (adapter, leakage, quality checks, feature engineering).
- `models/` contains model wrappers and selection logic.
- `monitoring/` provides drift and explainability primitives.
- `services/` manages reporting, registry, and export packaging.
- `pipelines/` defines preprocessing DAG nodes used by training/exported predictors.

## Data and Training Flow

1. CLI or MCP tool builds `ExperimentConfig`.
2. `Trainer` runs preprocessing DAG via `HamiltonExecutor`.
3. `DataPreparationService` performs adapter conversion, optional quality cleaning, leakage checks, and target transforms.
4. `ModelTrainer` executes baselines and selected models (sequential or concurrent), records metrics/events, and updates champion registry.
5. `DriftChecker` and `StateObserver` generate monitoring and workspace state artifacts.

## Persistence and Interfaces

- Experiment/events are appended to `workspace/experiments.jsonl`.
- Champion metadata is stored in `workspace/registry.json` with file locking.
- Export packages include `model.artifact`, `metadata.json`, `predictor.py`, and local preprocessing pipeline code.
- Optional integrations (`wandb`, `mlflow`, `llm`, `mcp`) are additive and controlled by extras/config.

## Guardrails

- Safe deserialization uses a restricted unpickler allowlist.
- HPO warmstart uses historical completion events with additive `params` fields.
- Metric directionality and registry promotion logic are centralized in `services/report_service.py`.
