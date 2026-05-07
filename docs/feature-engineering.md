# Automated Feature Engineering

Reference for the AFE pipeline: top-K feature selection, pairwise interaction discovery, and feature pruning.

---

## Overview

**Source:** `src/tabular_blueprint/data/feature_engine.py`

The AFE pipeline runs after initial preprocessing and baseline training:

1. **Top-K Selection** — rank features by permutation importance, keep top K
2. **Interaction Discovery** — test pairwise multiply/ratio features among top-K
3. **Pruning** (optional) — drop features below minimum importance threshold

**Orchestrator:** `src/tabular_blueprint/pipelines/nodes/feature_engineering.py` (Hamilton DAG node: `training_features__afe_enabled`)

---

## Permutation Importance Top-K Selection

**Source:** `src/tabular_blueprint/data/feature_engine.py:127`

**Function:** `extract_top_k_features(model_or_predictions, X, y, k, feature_names, task, random_seed)`

**Library:** `sklearn.inspection.permutation_importance`

**Algorithm:**
1. For each feature, shuffle its values `n_repeats=10` times
2. Measure the resulting drop in model performance
3. Mean importance = average drop across repeats
4. Return indices of top-K features by descending importance

**Scoring:** `roc_auc` (classification) or `r2` (regression)

**Mathematical Formulation:**

```
importance_j = 1/K Σ_k [score(fitted_model, X) - score(fitted_model, X_permuted_j_k)]
```

where `X_permuted_j_k` is the dataset with feature `j` shuffled in repeat `k`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 10 | Number of top features to select |
| `n_repeats` | 10 | Permutation repeats per feature |

---

## Pairwise Interaction Discovery

**Source:** `src/tabular_blueprint/data/feature_engine.py:156`

**Function:** `discover_interactions(X, y, top_k_indices, feature_names, task, lift_threshold, cv_folds, random_seed)`

**Library:** `sklearn.linear_model` (LogisticRegression / Ridge), `sklearn.model_selection.cross_val_score`

### Algorithm

1. Compute baseline CV score using a linear model on original features
2. For each pair `(i, j)` among top-K features:
   - Generate two interaction features:
     - **Multiply:** `x_i · x_j`
     - **Ratio:** `x_i / x_j` (safe division, `|b| > 1e-10` guard)
   - Augment `X` with the interaction feature
   - Compute augmented CV score with the same linear model
   - Compute lift: `lift = aug_mean - baseline_mean`
   - Keep the interaction if `lift > lift_threshold`
3. Return augmented feature matrix with all kept interactions

### Interaction Operations

| Operation | Implementation | Formula | Guard |
|-----------|---------------|---------|-------|
| `multiply` | `np.multiply(x_i, x_j)` | `f = x_i × x_j` | None |
| `ratio` | `_safe_ratio(x_i, x_j)` | `f = x_i / x_j` | Returns `0.0` where `|x_j| ≤ 1e-10`; returns `None` if all NaN/Inf |

### Lift Formula

```
lift = CV_score(X_augmented) - CV_score(X_original)
```

Only interactions with `lift > lift_threshold` are kept.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lift_threshold` | 0.01 | Minimum CV lift to keep an interaction |
| `cv_folds` | 3 | Folds for evaluating each candidate |

---

## Feature Pruning

**Source:** `src/tabular_blueprint/data/feature_engine.py:258`

**Function:** `prune_features(model, X, y, feature_names, min_importance, task, random_seed)`

**Library:** `sklearn.inspection.permutation_importance`

### Algorithm

1. Compute permutation importance for all features using the already-fitted GBDT model
2. Drop features with `mean_importance < min_importance`
3. Return pruned feature matrix

**Key Detail:** Reuses the GBDT model already fitted during training (no retraining needed).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_importance` | 0.001 | Minimum mean permutation importance to keep a feature |
| `n_repeats` | 10 | Permutation repeats |

---

## Full AFE Pipeline

**Source:** `src/tabular_blueprint/pipelines/nodes/feature_engineering.py` (Hamilton DAG node: `training_features__afe_enabled`)

```
1. Fit importance model (LightGBM/CatBoost)
2. extract_top_k_features() → top K indices
3. discover_interactions() → augmented X with interactions
4. Refit model on augmented features
5. (Optional) prune_features() → drop low-importance features
```

The pipeline is also available as Hamilton DAG nodes (`src/tabular_blueprint/pipelines/nodes/feature_engineering.py`) with config variants:
- `training_features__default` — no AFE, pass-through
- `training_features__afe_enabled` — runs full AFE pipeline
