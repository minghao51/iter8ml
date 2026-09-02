# Hyperparameter Optimization (HPO)

Reference for Optuna-based hyperparameter optimization, warmstarting, and search space refinement.

---

## Study Creation

**Source:** `src/iter8ml/engine/hpo.py:30`

**Function:** `create_study(model_name, direction, n_trials, pruner)`

**Library:** `optuna`

### Pruner Options

| Pruner | Value | Description |
|--------|-------|-------------|
| **MedianPruner** | `"median"` (default) | Prunes trials whose intermediate values are worse than the median of previous trials |
| **HyperbandPruner** | `"hyperband"` | Uses the Hyperband algorithm — allocates resources adaptively, early-stops poor configurations |
| **NopPruner** | `"nop"` | No pruning — every trial runs to completion |

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `direction` | `"maximize"` | Fallback direction for direct `create_study` calls; `optimize_model` always derives it from the primary metric (below) |
| `n_trials` | 50 | Number of HPO trials |
| `pruner` | `"median"` | Pruner strategy |

### Direction Resolution

**Source:** `src/iter8ml/services/reporting.py:43`

`optimize_model` resolves the study direction from the central direction
registry in `services/reporting.py` (`metric_higher_is_better`), never by
hardcoding it at call sites:

1. Primary metric = first entry of the `metrics` argument; when omitted, the
   evaluator's configured metric list is used (final fallback: the first
   returned score at trial time).
2. Lower-is-better metrics (`rmse`, `mae`, `mse`, `log_loss`, …) minimize;
   everything else maximizes.
3. The resolved direction is passed to both study creations (warmstarted and
   plain) and reported back in the result dict as `direction` +
   `primary_metric`.

So `iter8 hpo --task regression` (metrics `rmse, r2`) minimizes RMSE.

---

## Search Space Sampling

**Source:** `src/iter8ml/engine/hpo.py` (`_parse_trial_params`)

The `objective()` closure within `optimize_model()` samples parameters from the configured search space:

### Tuple Format

Each parameter in the search space dict is a tuple:

| Format | Meaning | Sampling |
|--------|---------|----------|
| `(low, high)` | Numeric range | `suggest_float` if any float bound, else `suggest_int` |
| `(low, high, "linear")` | Float range, linear scale | `suggest_float(low, high, log=False)` |
| `(low, high, "log")` | Float range, log scale | `suggest_float(low, high, log=True)` |

### Validation

- Bounds must be numeric
- `low < high` (strict inequality)
- Third element must be `"linear"` or `"log"`

---

## Full Optimization Loop

**Source:** `src/iter8ml/engine/hpo.py` (`optimize_model`)

**Function:** `optimize_model(model_cls, X, y, evaluator, model_name, n_trials, search_space, task, log_path, tracker, metrics)`

### Flow

1. Resolve the primary metric and study direction (see *Direction Resolution*)
2. Create or load warmstarted study
3. For each trial:
   - Sample hyperparameters from search space
   - Run cross-validation via `Evaluator.evaluate()`
   - Return the primary metric as trial value (first score when the primary is absent)
   - On failure: prune the trial
4. Log each trial to JSONL (if `log_path` provided)
5. Compute parameter importance via PedAnova
6. Return `best_params`, `best_value`, `n_trials`, `direction`, `primary_metric`, `param_importances`

### Thread Safety (OpenMP cap)

Every model import flows through `get_model_class()`
(`src/iter8ml/engine/models/factory.py`), which applies the OpenMP thread cap
(`HardwareProfile.configure_omp_threads()`, ADR-0004/0006) **before** the GBDT
module import can load libgomp. This makes HPO — and every other path that
resolves models via the factory — safe by default at the model factory; the
`Trainer.__init__` cap remains as a second layer.

On Intel hybrid CPUs (P+E cores, no SMT), the cap alone is not enough: libgomp's
default **active spin-wait** can live-lock GBDT training barriers across
heterogeneous cores. `configure_omp_threads()` therefore also sets
`OMP_WAIT_POLICY=passive`, and the GBDT wrappers pin their per-library thread
counts (`num_threads` / `nthread` / `thread_count`) to the same cap — user
overrides still win.

### HPO Trial Logging

Each completed trial is logged as a JSONL event:

```json
{
  "event": "hpo_trial_completed",
  "run_id": "hpo_catboost",
  "model": "catboost",
  "params": {"depth": 8, "learning_rate": 0.03},
  "cv_scores": {"roc_auc": 0.85}
}
```

---

## Warmstarted HPO

**Source:** `src/iter8ml/engine/hpo_warmstart.py:90`

**Function:** `create_warmstarted_study(model_name, direction, log_path, n_trials, pruner)`

### Concept

Pre-populates a new Optuna study with historical trial data from previous experiment runs. This gives the optimizer a head start by seeding it with known parameter-score pairs.

### Flow

1. Load events from `experiments.jsonl`
2. Filter for `model_completed` and `hpo_trial_completed` events matching `model_name`
3. For each event:
   - Extract `params` and primary `cv_scores` value
   - Infer Optuna distributions from parameter names and values
   - Create a `COMPLETE` trial and add it to the study
4. Return the pre-warmed study

### Distribution Inference

**Source:** `src/iter8ml/engine/hpo_warmstart.py:41`

**Function:** `_infer_distribution(name, value)`

Uses naming conventions to choose the right distribution:

| Name Pattern | Distribution | Range |
|-------------|-------------|-------|
| `*n_estimators*`, `*iterations*` | `IntDistribution` | (50, 5000) |
| `*depth*`, `*max_depth*` | `IntDistribution` | (2, 15) |
| `*num_leaves*` | `IntDistribution` | (8, 256) |
| `*batch_size*` | `IntDistribution` | (32, 1024) |
| `*n_epochs*` | `IntDistribution` | (10, 200) |
| `*lr*`, `*learning_rate*` | `FloatDistribution` | (1e-5, 0.5, log=True) |
| `*dropout*` | `FloatDistribution` | (0.0, 0.5) |
| `*subsample*`, `*colsample*` | `FloatDistribution` | (0.4, 1.0) |
| `bool` values | `CategoricalDistribution` | [True, False] |
| Generic `int` | `IntDistribution` | (value×0.5, value×2) |
| Generic `float` | `FloatDistribution` | (value×0.5, value×2) |

---

## Parameter Importance Analysis

**Source:** `src/iter8ml/engine/hpo_importance.py:31`

**Function:** `compute_param_importance(study, evaluator_class)`

**Library:** `optuna.importance.PedAnovaImportanceEvaluator`

### Algorithm: PedAnova

Uses a permutation-based ANOVA approach to rank hyperparameters by their impact on the objective:

1. For each hyperparameter, permute its values across trials
2. Measure how much the objective predictions change
3. Normalize to produce importance scores (summing to 1.0)

This is more robust than frequency-based methods and works well with small trial counts.

**Output:** `ImportanceReport` — ranked list of `(param_name, importance)` pairs

---

## Search Space Refinement

**Source:** `src/iter8ml/engine/hpo_importance.py:75`

**Function:** `suggest_refined_space(study, original_space, top_k, importance_threshold, expansion_factor)`

### Algorithm

Narrows search bounds toward high-performing regions discovered during HPO:

1. Compute parameter importance via PedAnova
2. For each important parameter (above `importance_threshold`):
   - Collect all trial values from the study
   - Compute Q25 and Q75 of those values
   - Expand the interquartile range by `expansion_factor`
   - Clamp to original bounds
3. Low-importance parameters keep their original bounds

**Formula:**

```
span = Q75 - Q25
new_low  = max(original_low,  Q25 - expansion_factor × span)
new_high = min(original_high, Q75 + expansion_factor × span)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | `None` | Only refine top K params (None = all above threshold) |
| `importance_threshold` | 0.01 | Minimum importance to trigger refinement |
| `expansion_factor` | 1.3 | 30% wider than IQR — balances exploitation and exploration |

---

## Setup Helper

**Source:** `src/iter8ml/engine/hpo.py` (`setup_hpo_components`)

**Function:** `setup_hpo_components(data_path, target_col, task, model, cv_folds=None, metrics=None, random_seed=None, ignore_cols=None, positive_class=None)`

Shared setup for CLI and MCP HPO entry points:

1. Loads data via `load_data()`
2. Routes the frame through the same prep chain as training
   (`PipelineExecutor(mode=PipelineMode.HPO).run_prep(...)`): `ignore_cols`
   filter → `positive_class` orientation → null fill → date decomposition →
   categorical encoding → target validation. String categoricals reach
   `DataAdapter` as numeric codes — raw strings would crash LightGBM/XGBoost
   constructors. Pass `ignore_cols`/`positive_class` from the experiment
   config so HPO scores the same feature set and class orientation as
   `iter8 run` (quality-audit and feature-engineering steps remain
   training-path-only).
3. Converts to numpy via `DataAdapter()` (features **and** encoded target)
4. Creates an `Evaluator` (defaults, or the `cv_folds`/`metrics`/`random_seed`
   overrides passed by the CLI `--config` path, so HPO folds use the same
   seed and fold count as `iter8 run`)
5. Resolves the model's HPO search space from `ModelConfigs`

Returns `(X, y, evaluator, search_space)`.

## Failure Semantics

Every trial evaluation exception is converted to `optuna.TrialPruned` (the
study keeps going — one bad hyperparameter region must not kill the search).
A study that completes fewer than `min(n_trials, max(3, n_trials // 10))`
trials raises `ValueError` surfacing the first trial's underlying exception
instead of returning: an all-pruned study has no meaningful `best_value`, and
crowning a winner over a tiny survivor set would mislead. The CLI reports
this as `Error: ...` and exits 1. Note: warmstart-injected historical trials
count toward the threshold — they are real prior evaluations, so a study can
complete on injected evidence alone.

## CLI

`iter8 hpo` accepts either explicit flags or an experiment config:

```bash
# explicit flags (defaults for the rest)
iter8 hpo --data train.csv --target y --model lightgbm --trials 50

# config-driven: reuses task, target_col, data_path, cv_folds, metrics,
# primary_metric (optimized first), random_seed, ignore_cols, positive_class
# and per-model model_overrides (as fixed params) from the config file;
# explicit CLI flags override the config values
iter8 hpo --config config.yaml --trials 100
```

The resolved settings are echoed before the study starts.
