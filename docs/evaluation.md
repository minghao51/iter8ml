# Evaluation & Metrics

Reference for cross-validation strategies, evaluation metrics, lift computation, and probability calibration.

---

## Cross-Validation Strategies

**Source:** `src/tabular_blueprint/engine/evaluator.py:34`

**Function:** `get_cv_split(strategy, n_splits)`

| Strategy | Enum | Splitter | Description |
|----------|------|----------|-------------|
| `"kfold"` | `CVStrategy.KFOLD` | `sklearn.model_selection.KFold` | Random splits, `shuffle=True`, `random_state=42` |
| `"stratified"` | `CVStrategy.STRATIFIED` | `sklearn.model_selection.StratifiedKFold` | Preserves class proportions per fold |
| `"timeseries"` | `CVStrategy.TIMESERIES` | `sklearn.model_selection.TimeSeriesSplit` | Temporal splits — fold i trains on [0..i], validates on [i+1] |

**Default:** `n_splits=5`

---

## Evaluation Pipeline

**Source:** `src/tabular_blueprint/engine/evaluator.py:46`

**Class:** `Evaluator`

The `evaluate()` method runs K-fold cross-validation and computes all configured metrics:

1. Creates a fresh model instance per fold (prevents state leakage)
2. Fits on train split, predicts on validation split
3. Computes each configured metric
4. Aggregates fold scores via `np.mean()`

For probability-based metrics (`roc_auc`, `log_loss`), `predict_proba()` is called. Otherwise, `predict()` is used.

---

## Metrics Registry

**Source:** `src/tabular_blueprint/engine/evaluator.py:19`

### Classification Metrics

| Metric | Key | Library | Formula |
|--------|-----|---------|---------|
| **ROC AUC** | `roc_auc` | `sklearn.metrics.roc_auc_score` | Area under the Receiver Operating Characteristic curve: `AUC = P(score(positive) > score(negative))` |
| **F1 Macro** | `f1_macro` | `sklearn.metrics.f1_score(average="macro")` | Harmonic mean of precision and recall, averaged across classes: `F1 = 2·(P·R)/(P+R)` |
| **Accuracy** | `accuracy` | `sklearn.metrics.accuracy_score` | Fraction of correct predictions: `Acc = (TP+TN) / N` |
| **Log Loss** | `log_loss` | `sklearn.metrics.log_loss` | Cross-entropy loss: `L = -1/N Σ Σ y_{ic} · log(p_{ic})` |

### Regression Metrics

| Metric | Key | Library | Formula |
|--------|-----|---------|---------|
| **RMSE** | `rmse` | `numpy.sqrt(sklearn.metrics.mean_squared_error)` | `RMSE = √(1/N Σ(y_i - ŷ_i)²)` |
| **MAE** | `mae` | `sklearn.metrics.mean_absolute_error` | `MAE = 1/N Σ|y_i - ŷ_i|` |
| **R²** | `r2` | `sklearn.metrics.r2_score` | Coefficient of determination: `R² = 1 - SS_res / SS_tot = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²` |

**Lower-is-better metrics** (`rmse`, `mae`, `log_loss`): positive lift means the model improved over baseline.

---

## Lift Computation

**Source:** `src/tabular_blueprint/engine/evaluator.py:132`

**Method:** `Evaluator.compute_lift(model_scores, baseline_scores, metric_name)`

**Formula:**

```
lift = (model_score - baseline_score) / |baseline_score|
```

For lower-is-better metrics, the sign is flipped:

```
lift = (baseline_score - model_score) / |baseline_score|
```

Returns a fraction (e.g., `0.15` = 15% lift). Returns `0.0` if baseline is zero.

---

## Probability Calibration

**Source:** `src/tabular_blueprint/engine/calibration.py:17`

**Class:** `CalibratedModel`

Wraps any model with probability calibration. Applied post-training to improve the reliability of `predict_proba()` outputs.

### Platt Scaling (`method="platt"`)

**Library:** `sklearn.calibration.CalibratedClassifierCV(method="sigmoid")`

**Mathematical Formulation:**
Fits a logistic regression on the model's raw scores:

```
P(y=1|x) = 1 / (1 + exp(A · f(x) + B))
```

where `f(x)` is the uncalibrated model output and `A, B` are fitted via logistic regression on the validation set.

### Isotonic Regression (`method="isotonic"`)

**Library:** `sklearn.calibration.CalibratedClassifierCV(method="isotonic")`

**Mathematical Formulation:**
Fits a non-parametric, monotonically increasing step function that maps raw scores to calibrated probabilities. More flexible than Platt scaling but requires more data:

```
P(y=1|x) = isotonic(f(x))
```

where `isotonic()` is a piecewise-constant monotonically increasing function fitted via the pool-adjacent-violators algorithm (PAVA).

### Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"none"` | `"platt"`, `"isotonic"`, or `"none"` |
| `cv_folds` | `int` | 3 | StratifiedKFold folds for calibration fitting |

### Behavior
- If `method="none"` or the base model lacks `predict_proba()`, calibration is skipped and the base model is used as-is.
- Calibration uses `StratifiedKFold` internally to avoid overfitting the calibration map.
