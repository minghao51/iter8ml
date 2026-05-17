# iter8ml — Current State

Last updated: 2026-05-16

## What's Implemented

| Feature | Backend | Frontend/CLI |
|---------|---------|--------------|
| Data loading (CSV, Parquet, SQLite) | Full (`data/loader.py`) | CLI via `--data` flag |
| Data preprocessing (null fill, date decomp, encoding) | Full (`engine/pipelines/nodes/prep.py`) | N/A (pipeline node) |
| Data quality audit (Cleanlab) | Full (`data/quality.py`) | Config-gated |
| Leakage detection (permutation importance) | Full (`data/leakage.py`) | Config-gated |
| Target transform (log1p, box-cox, yeo-johnson) | Full (`data/features.py:57-104`) | Config-gated |
| Model: CatBoost | Full (`engine/models/catboost_model.py`) | Auto/routed |
| Model: LightGBM | Full (`engine/models/lightgbm_model.py`) | Auto/routed |
| Model: XGBoost | Full (`engine/models/xgboost_model.py`) | Auto/routed |
| Model: TabPFN v2 | Full (`engine/models/tabpfn_model.py`) | Auto/routed (GPU-gated) |
| Model: FT-Transformer | Full (`engine/models/ft_transformer.py`) | VRAM-gated (>12GB) |
| Model: TabNet | Full (`engine/models/tabnet_model.py`) | VRAM-gated (>8GB) |
| Model: Naive/Linear baselines | Full (`engine/models/baselines.py`) | Auto-included |
| Model selection (hardware-aware routing) | Full (`engine/models/selector.py`) | Automatic |
| Model factory (plugin discovery) | Full (`engine/models/factory.py`) | Entry-point based |
| Cross-validation (KFold, Stratified, TimeSeries) | Full (`engine/evaluator.py`) | Configurable |
| Probability calibration (Platt, Isotonic) | Full (`engine/calibration.py`) | Config-gated |
| HPO (Optuna) | Full (`engine/hpo.py`) | CLI `hpo` command |
| HPO warmstart (historical trial injection) | Full (`engine/hpo_warmstart.py`) | CLI `--log` flag |
| HPO param importance (PedAnova) | Full (`engine/hpo_importance.py`) | Automatic |
| AFE (interaction discovery + pruning) | Full (`data/features.py:204-367`) | `feature_strategy=afe` |
| Entity embedding (high-cardinality) | Full (`data/embedding.py`) | `feature_strategy=embedding` |
| DAE embedding | Full (`data/embedding.py:269-311`) | `embedding_method=autoencoder` |
| Drift: KS/Chi2 | Full (`analysis/drift.py`) | CLI `drift --method ks` |
| Drift: PSI | Full (`analysis/psi.py`) | CLI `drift --method psi` |
| Drift: Domain classifier | Full (`analysis/domain_classifier.py`) | CLI `drift --method domain` |
| SHAP explainability | Full (`analysis/explainability.py`) | `shap_enabled=True` |
| Model registry (file-locking) | Full (`services/registry.py`) | CLI `registry show` |
| Model export (portable package) | Full (`services/export.py`) | CLI `export` command |
| JSONL experiment tracking + rotation | Full (`engine/tracker.py`) | Default tracker |
| W&B tracking | Full (`engine/tracker.py:98-125`) | `tracker=wanDB` |
| MLflow tracking | Full (`engine/tracker.py:128-160`) | `tracker=mlflow` |
| LLM agent (SHAP + performance commentary) | Full (`services/llm.py`) | `llm_enabled=True` |
| MCP server (8 tools for LLM agents) | Full (`services/mcp.py`) | Lazy-loaded |
| State observer (current_state.md) | Full (`engine/state_observer.py`) | CLI `state` |
| Report service (leaderboard) | Full (`services/reporting.py`) | CLI `leaderboard` |
| CLI: init, hardware, run, hpo, drift, state, leaderboard, diff, export, registry | Full (`cli/`) | Typer-based |
| ExperimentSession (Python API) | Full (`session.py`) | Programmatic |
| Preprocessing cache | Full (`data/cache.py`) | Workspace-based |
| Safe pickle (RestrictedUnpickler) | Full (`utils/io.py:74-97`) | Internal |
| Pipeline DAG (Hamilton) | Full (`engine/pipelines/`) | Config-driven |

## Stubbed / Unimplemented

No `NotImplementedError` or 501 responses found in the codebase. All features are fully implemented.

However, there are no-op `pass` bodies in abstract/protocol methods that are expected to be overridden:

- `engine/models/factory.py:30` — `_discover_models()` fallback `pass` (expected; entry point not found)
- `engine/models/gbdt_base.py:27,32,72,77,87,98,104` — Abstract method `pass` bodies (required by ABC pattern)
- `engine/models/tabpfn_model.py:12,16` — Custom exception class `pass` bodies (empty by design)
- `engine/models/tabpfn_model.py:35` — `except ImportError: pass` (graceful CPU fallback)
- `engine/models/ft_transformer.py:39` — `_ModuleBase` placeholder `pass` when torch absent
- `engine/pipelines/hooks/tracking_hook.py:19,28` — `run_before_node_execution` / `run_after_node_execution` are no-op hooks (by design)

## Known Bugs

| Severity | Issue | Location |
|----------|-------|----------|
| Low | `cli/run.py:101` — `results.items()` contains `ModelResult`-like dicts with nested structure, not simple scores; output is messy | `cli/run.py:101` |

## Security Concerns

| Severity | Issue | Location |
|----------|-------|----------|
| Medium | `safe_dump` uses `pickle.dump` with no MAC/signature — tampered files pass `RestrictedUnpickler` if classes are allowlisted | `utils/io.py:100-103` |
| Medium | `load_sqlite` query validation strips SQL keywords from uppercase, but creative bypass is possible via encoding tricks or subqueries | `data/loader.py:76-88` |
| Medium | `Config.from_file` with `allow_unsafe_python=True` executes arbitrary `.py` files via `exec_module` | `config.py:281-295` |
| Low | `baselines.py:52` — `np.load(path + ".npz", allow_pickle=False)` is safe, but `.npz` extension must be manually appended by caller — mismatch risk | `engine/models/baselines.py:52` |
| Low | `ft_transformer.py:166` — `torch.load(path, ..., weights_only=True)` is safe, but only enforced in this one model | `engine/models/ft_transformer.py:166` |
| Low | `__init__.py:7` — Bare `except Exception` silently swallows all errors during version detection | `__init__.py:7` |

## Performance Issues

| Issue | Location |
|-------|----------|
| `detect_leakage` runs full cross-validation per feature (O(n_features * cv_folds) model fits) — extremely slow for wide datasets | `data/leakage.py:52-83` |
| `discover_interactions` runs cross-validation for every candidate pair × operation — O(top_k^2 * cv_folds) evaluations | `data/features.py:234-261` |
| `extract_top_k_features` uses `permutation_importance` with `n_repeats=10` — expensive for large datasets | `data/features.py:189-201` |
| `prune_features` runs a second `permutation_importance` pass — redundant if already computed during AFE | `data/features.py:348-350` |
| `audit_data_quality` runs `cross_val_predict` on `LogisticRegression` — no GPU or parallel support | `data/quality.py:43` |
| `DomainClassifierDriftDetector.detect` runs `cross_val_score` on combined ref+live data — memory scales with both datasets | `analysis/domain_classifier.py:55-69` |
| `_train_one` trains models sequentially; `max_workers` config field exists but is never used for parallelism | `engine/pipelines/nodes/train.py:150-221` |
| `EmbeddingEngine._train_entity` and `_train_autoencoder` hardcode `device="cpu"` even when GPU is available | `data/embedding.py:232,288` |
| `_build_model` in `TabNetModel` creates full `TabularModel` config on every `fit()` call — wasteful if called repeatedly | `engine/models/tabnet_model.py:24-57` |

## Maintenance Issues

| Issue | Detail |
|-------|--------|
| `max_workers` config field unused | `config.py:163` defines `max_workers` but `train.py` trains sequentially — dead config |
| Broad `except Exception` handlers | `__init__.py:7`, `services/mcp.py:188`, `engine/hpo.py:232,264`, `services/llm.py:100`, `engine/models/catboost_model.py:30`, `exceptions.py:60` — all catch `Exception` broadly |
| `TrackingHook` before/after hooks are no-ops | `engine/pipelines/hooks/tracking_hook.py:13-19,21-28` — registered but do nothing, only error hook logs |
| `ExperimentConfig.__getattr__` can confuse Pydantic | `config.py:183-187` — custom `__getattr__` for flat delegates may cause issues with `hasattr()` checks |
| `pyproject.toml` has `[train]` extras but models are core deps | `catboost`, `lightgbm`, `xgboost`, `optuna` are in `[train]` optional deps but imported unconditionally in model wrappers |
| `Tracker` is a Protocol but `JSONLTracker` doesn't inherit it | `engine/tracker.py:13,23` — structural subtyping works but explicit inheritance would clarify intent |
