# Hyperparameter Optimization (HPO)

Reference for Optuna-based hyperparameter optimization, warmstarting, and search space refinement.

---

## Study Creation

**Source:** `src/iter8ml/engine/hpo.py:15`

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
| `direction` | `"maximize"` | Optimize for higher metric (ROC AUC, R²) |
| `n_trials` | 50 | Number of HPO trials |
| `pruner` | `"median"` | Pruner strategy |

---

## Search Space Sampling

**Source:** `src/iter8ml/engine/hpo.py:178`

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

**Source:** `src/iter8ml/engine/hpo.py:87`

**Function:** `optimize_model(model_cls, X, y, evaluator, model_name, n_trials, search_space, task, log_path)`

### Flow

1. Create or load warmstarted study
2. For each trial:
   - Sample hyperparameters from search space
   - Run cross-validation via `Evaluator.evaluate()`
   - Return primary metric as trial value
   - On failure: prune the trial
3. Log each trial to JSONL (if `log_path` provided)
4. Compute parameter importance via PedAnova
5. Return `best_params`, `best_value`, `n_trials`, `param_importances`

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

**Source:** `src/iter8ml/engine/hpo.py:43`

**Function:** `setup_hpo_components(data_path, target_col, task, model)`

Shared setup for CLI and MCP HPO entry points:

1. Loads data via `load_data()`
2. Converts to numpy via `DataAdapter()`
3. Creates an `Evaluator` with default config
4. Resolves the model's HPO search space from `ModelConfigs`

Returns `(X, y, evaluator, search_space)`.
