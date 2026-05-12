# Explainability

Reference for SHAP-based model explanations: global feature importance and visualization.

---

## SHAP Explainer

**Source:** `src/iter8ml/monitoring/explainability.py:24`

**Class:** `Explainer`

### Method Selection

**Source:** `monitoring/explainability.py:76`

The explainer automatically selects the appropriate SHAP method based on the model type:

| Model Type | SHAP Method | Description |
|------------|-------------|-------------|
| LightGBM, XGBoost, CatBoost, GBDT | **`shap.TreeExplainer`** | Exact Shapley values for tree models in O(TLD²) time |
| All others | **`shap.KernelExplainer`** | Model-agnostic approximation using a background sample |

**Detection Logic:** Checks if `model_type` or `model_name` contains any of: `"lgbm"`, `"lightgbm"`, `"xgboost"`, `"xgb"`, `"catboost"`, `"gbdt"`.

For `KernelExplainer`, a background dataset of `min(100, n_samples)` rows is sampled via `shap.sample()`.

---

## Mathematical Formulation

### Shapley Values

SHAP (SHapley Additive exPlanations) computes feature attributions based on Shapley values from cooperative game theory:

```
φ_i = Σ_{S⊆F\{i}} (|S|!(|F|-|S|-1)!) / |F|! × [f(S∪{i}) - f(S)]
```

where:
- `φ_i` is the Shapley value for feature `i`
- `F` is the set of all features
- `S` is a subset of features not containing `i`
- `f(S)` is the model prediction using only features in `S`

### TreeExplainer

For tree ensembles, TreeExplainer computes exact Shapley values in polynomial time by traversing all tree paths. The algorithm exploits the tree structure to avoid the exponential enumeration of feature subsets.

### KernelExplainer

Uses a weighted linear regression on a synthetic dataset of feature coalitions:

```
minimize Σ_{z∈Z} π(z) × (f(h_x(z)) - g(z))²
```

where `π(z)` is the SHAP kernel weight and `h_x(z)` maps binary coalitions to actual feature values.

---

## Global Feature Importance

**Source:** `monitoring/explainability.py:35`

**Method:** `Explainer.explain(X, run_id, max_display, generate_plots)`

### Computation

1. Compute SHAP values for all samples: `explainer(X)`
2. Handle multi-output models: if `values.ndim == 3`, aggregate across classes: `mean(|φ|)` over `(samples, classes)` → per-feature scalar
3. For 2D output: `mean(|φ|)` over samples → per-feature scalar
4. Rank features by descending mean absolute SHAP value

### Output

`SHAPExplanationResult`:

| Field | Description |
|-------|-------------|
| `model_name` | Name of the explained model |
| `n_features` | Total number of features |
| `top_features` | Ranked list of `FeatureImportance(feature_name, importance)` up to `max_display` |
| `plot_paths` | File paths to generated plots |

---

## Visualization

**Source:** `monitoring/explainability.py:99`

**Library:** `shap`, `matplotlib`

### Beeswarm Plot

Shows the distribution of SHAP values for each feature across all samples. Features are ordered by importance (top = most important). Color represents feature value (red = high, blue = low).

```
workspace/artifacts/shap_{run_id}/beeswarm.png
```

### Dependence Plots

Scatter plots of SHAP value vs feature value for the top 5 features, showing the marginal effect of each feature on the prediction.

```
workspace/artifacts/shap_{run_id}/dependence_{i}.png
```

| Plot | DPI | Format |
|------|-----|--------|
| Beeswarm | 150 | PNG |
| Dependence | 150 | PNG (top 5 features) |

---

## Orchestration

**Source:** `src/iter8ml/monitoring/explainability.py:28`

**Class:** `Explainer`

The explainer orchestrates SHAP explanation and logs results to the experiment tracker. Called automatically during the training pipeline when SHAP is enabled in the experiment config.
