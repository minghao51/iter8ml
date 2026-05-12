# Structure

## Root Directory Layout

```
iter8ml/
├── src/
│   └── iter8ml/          # Main package (src layout)
├── tests/
│   ├── conftest.py
│   ├── unit/                       # Fast, isolated tests
│   ├── integration/                # Slow tests requiring services
│   └── e2e/                        # Full workflow smoke tests
├── notebooks/                      # Marimo notebooks
├── docs/                           # MkDocs documentation
├── scripts/                        # Build/deploy scripts
├── benchmarks/                     # Performance benchmarks
├── examples/                       # Usage examples
├── workspace/                      # Runtime data (experiments.jsonl, registry.json, artifacts/)
├── .planning/                      # Architecture documentation
│   └── codebase/
├── .github/                        # CI/CD workflows
├── .devcontainer/                  # Dev container config
├── pyproject.toml                  # Project metadata, deps, tool config
├── mkdocs.yml                      # Documentation config
├── Dockerfile
├── docker-compose.yml
├── AGENTS.md                       # Agent instructions
├── ARCHITECTURE.md                 # High-level architecture docs
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CLAUDE.md
├── README.md
├── docs/
│   └── technical_roadmap.md
└── uv.lock
```

## Source Code: `src/iter8ml/`

### Top-Level Modules

| File | Size | Purpose |
|---|---|---|
| `__init__.py` | 4 lines | Package marker, `__all__ = []`, `__typed__ = True` |
| `cli.py` | 477 lines | Typer CLI entry point (10 commands) |
| `config.py` | 186 lines | `ExperimentConfig`, `HardwareProfile` |
| `constants.py` | 75 lines | Enums: `TaskType`, `CVStrategy`, `ModelName`, `EmbeddingMethod`, `TrackerType` |
| `exceptions.py` | 74 lines | Exception hierarchy + `track_errors` decorator |

### `data/` — Data Loading and Transformation

| File | Purpose |
|---|---|
| `__init__.py` | Exports: `augment_with_embeddings`, `detect_high_cardinality_columns`, `extract_cat_codes`, `get_data_hash`, `load_csv`, `load_parquet`, `load_sqlite` |
| `loaders.py` | CSV/Parquet/SQLite loading, data hashing |
| `adapter.py` | `DataAdapter`: Polars -> numpy conversion |
| `cache.py` | `PreprocessingCache`: hash-based preprocessing cache |
| `embedding_engine.py` | Entity embedding and autoencoder utilities |
| `feature_engine.py` | AFE: interaction discovery, feature extraction, pruning, target transform |
| `leakage.py` | `detect_leakage()`, `LeakageReport` for target leakage detection |
| `quality.py` | `audit_data_quality()`, `clean_noise()` |

### `engine/` — Core Orchestration

| File | Purpose |
|---|---|
| `__init__.py` | Exports: `Evaluator`, `JSONLTracker`, `Trainer` |
| `trainer.py` | `Trainer`: main orchestrator for Hamilton DAG training execution |
| `evaluator.py` | `Evaluator`: cross-validation evaluation loop |
| `tracker.py` | `Tracker` protocol + `JSONLTracker` (default), `WandbTracker`, `MLflowTracker` |
| `hpo.py` | `optimize_model()`, `setup_hpo_components()` for Optuna HPO |
| `hpo_warmstart.py` | Historical trial injection from JSONL log |
| `hpo_importance.py` | Parameter importance analysis (PedAnova) |
| `calibration.py` | `CalibratedModel`: Platt/Isotonic calibration wrapper |
| `state_observer.py` | `StateObserver`: generates `current_state.md` from logs + registry |

### `pipelines/` — Hamilton DAG Execution

| File | Purpose |
|---|---|
| `__init__.py` | Exports: `PipelineExecutor`, `PipelineMode`, `visualize_pipeline` |
| `executor.py` | `PipelineExecutor`: Hamilton driver builder with `PipelineMode` enum (TRAINING/DRIFT/EXPORT/HPO/INFERENCE) |
| `preprocessing.py` | Re-exports all 9 preprocessing node functions |
| `hooks/__init__.py` | Empty |
| `hooks/tracking_hook.py` | `TrackingHook`: `NodeExecutionHook` adapter for Tracker protocol |
| `nodes/__init__.py` | Empty (namespace package) |
| `nodes/preprocessing.py` | 9 nodes: null imputation, date decomposition, categorical encoding |
| `nodes/data_preparation.py` | 7 nodes: target validation, quality cleaning, adapter, leakage, target transform |
| `nodes/model_selection.py` | 1 node: auto/explicit model list |
| `nodes/baselines.py` | 2 nodes: naive + linear baseline evaluation |
| `nodes/feature_engineering.py` | Conditional nodes: passthrough, AFE, embedding (via `@config.when`) |
| `nodes/model_training.py` | 1 node: training loop -> `list[ModelResult]` |
| `nodes/state_generation.py` | 1 node: terminal -> `TrainingState` (leaderboard + registry update) |
| `nodes/drift_detection.py` | Conditional nodes: PSI, domain classifier, both (via `@config.when`) |

### `models/` — Model Implementations

| File | Purpose |
|---|---|
| `__init__.py` | Exports: `AbstractModel`, `ModelSelector`, `available_model_names`, `get_model_class`, `validate_model_name` |
| `base.py` | `AbstractModel` Protocol (structural subtyping) |
| `factory.py` | String-keyed model registry with lazy imports (`_MODEL_REGISTRY`) |
| `selector.py` | `ModelSelector`: hardware/data-size-aware model routing |
| `baselines.py` | `NaiveBaseline`, `LinearBaseline` |
| `gbdt_base.py` | Shared GBDT base class |
| `model_configs.py` | Hyperparameter configurations |
| `conventional/__init__.py` | Empty |
| `conventional/catboost_model.py` | CatBoostModel wrapper |
| `conventional/lightgbm_model.py` | LightGBMModel wrapper |
| `conventional/xgboost_model.py` | XGBoostModel wrapper |
| `deep/__init__.py` | Empty |
| `deep/ft_transformer.py` | FTTransformerModel (PyTorch) |
| `deep/tabnet_model.py` | TabNetModel (pytorch-tabular) |
| `deep/sparse_embedder.py` | SparseEmbedder utility |
| `deep/text_encoder.py` | TextEncoder utility |
| `tabular_foundation/__init__.py` | Empty |
| `tabular_foundation/tabpfn_model.py` | TabPFNModel (TabPFN v2) |

### `monitoring/` — Drift and Explainability

| File | Purpose |
|---|---|
| `__init__.py` | Module marker |
| `drift.py` | `DriftDetector`: KS test (numeric) + chi2 (categorical) |
| `psi_drift.py` | `PSIDriftDetector`: Population Stability Index |
| `domain_classifier.py` | `DomainClassifierDriftDetector`: train classifier to detect drift |
| `explainability.py` | SHAP-based explainer |

### `services/` — Registry, Reports, Export

| File | Purpose |
|---|---|
| `__init__.py` | Exports: `ExperimentReport`, `LeaderboardEntry`, `PromotionResult`, `RegistryService`, `ReportService` |
| `registry_service.py` | `RegistryService`: file-locked JSON registry, atomic saves |
| `report_service.py` | `ReportService`: leaderboard building, metric direction logic, markdown/console formatting |
| `export_service.py` | `ExportService`: packages champion model + preprocessing + predictor |

### `llm/` — LLM Integration

| File | Purpose |
|---|---|
| `__init__.py` | `TabularAgent`, `LLMCommentary`, `LLMAgentConfig` — natural language explanations via litellm |

### `mcp/` — MCP Server

| File | Purpose |
|---|---|
| `__init__.py` | Module marker |
| `tools.py` | FastMCP server with 9 atomic tools for LLM agents |

### `utils/` — Shared Utilities

| File | Purpose |
|---|---|
| `__init__.py` | Empty |
| `jsonl.py` | `load_events()`, `iter_events()` |
| `safe_pickle.py` | `RestrictedUnpickler`, `safe_load()`, `safe_load_file()`, `safe_dump()` |

## Test Structure: `tests/`

Tests mirror the source structure with unit/integration/e2e divisions:

```
tests/
├── conftest.py                    # Shared fixtures: classification_data, regression_data, tmp_workspace; auto-tagging by path
├── unit/
│   ├── test_adapter.py
│   ├── test_baselines.py
│   ├── test_calibration.py
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_data_prep_nodes.py
│   ├── test_domain_classifier.py
│   ├── test_drift.py
│   ├── test_drift_nodes.py
│   ├── test_embedding_engine.py
│   ├── test_embedding_trainer.py
│   ├── test_evaluator.py
│   ├── test_exceptions.py
│   ├── test_export_service.py
│   ├── test_feature_engine.py
│   ├── test_ft_transformer.py
│   ├── test_hpo_importance.py
│   ├── test_hpo_unit.py
│   ├── test_hpo_warmstart.py
│   ├── test_jsonl.py
│   ├── test_leakage.py
│   ├── test_llm_agent.py
│   ├── test_loaders.py
│   ├── test_mcp_tools.py
│   ├── test_model_factory.py
│   ├── test_model_selector.py
│   ├── test_pipeline_executor.py
│   ├── test_processors.py
│   ├── test_psi_drift.py
│   ├── test_quality.py
│   ├── test_registry_service.py
│   ├── test_report_service.py
│   ├── test_safe_pickle.py
│   ├── test_sparse_embedder.py
│   ├── test_state_observer.py
│   ├── test_tabpfn.py
│   ├── test_tabpfn_guardrails.py
│   ├── test_tracker_rotation.py
│   ├── test_trainer.py
│   └── test_training_nodes.py
├── integration/
│   ├── conftest.py
│   ├── test_export_package.py
│   ├── test_full_pipeline.py
│   ├── test_gdbt_models.py
│   ├── test_hpo.py
│   ├── test_model_selection.py
│   └── test_registry_and_drift.py
└── e2e/
    └── test_smoke.py
```

## Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase` — `ExperimentConfig`, `PipelineExecutor`, `ModelSelector`
- **Functions**: `snake_case` — `load_data()`, `detect_leakage()`, `metric_value_is_better()`
- **Test files**: `test_<module>.py` — mirrors source module name
- **Test classes**: not used; standalone test functions with `test_` prefix
- **Node functions**: descriptive DAG node names — `processed_dataframe`, `training_state`, `data_prep_result`

## Directory Symmetry

| Source | Tests |
|---|---|
| `data/adapter.py` | `tests/unit/test_adapter.py` |
| `engine/trainer.py` | `tests/unit/test_trainer.py` |
| `models/factory.py` | `tests/unit/test_model_factory.py` |
| `pipelines/executor.py` | `tests/unit/test_pipeline_executor.py` |
| `services/registry_service.py` | `tests/unit/test_registry_service.py` |
| `monitoring/drift.py` | `tests/unit/test_drift.py` |
| `engine/hpo.py` | `tests/unit/test_hpo_unit.py`, `tests/integration/test_hpo.py` |

Integration tests test cross-module workflows (full pipeline, model training, export, registry+drift).
