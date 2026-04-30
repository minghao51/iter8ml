# Preprocessing

Reference for the preprocessing pipeline, target transformation, data quality, leakage detection, and format conversion.

---

## Preprocessing Pipeline (Hamilton DAG Nodes)

**Source:** `src/tabular_blueprint/pipelines/nodes/preprocessing.py`

The preprocessing pipeline is implemented as Hamilton DAG nodes that execute in dependency order. All operations use Polars for lazy, columnar processing.

### DAG Node Sequence

```
raw_dataframe → [numeric_columns, categorical_columns, date_columns]
                  ↓                    ↓
           fill_nulls_numeric → fill_nulls_categorical
                  ↓                    ↓
                  └──── null_filled_df ────┘
                           ↓
                    decomposed_dates_df
                           ↓
                       encoded_df
                           ↓
                    processed_dataframe
```

### Column Detection

| Node | Input | Output | Logic |
|------|-------|--------|-------|
| `numeric_columns` | `raw_dataframe` | `list[str]` | `cs.numeric()` selector |
| `categorical_columns` | `raw_dataframe` | `list[str]` | `cs.categorical() \| cs.string()` |
| `date_columns` | `raw_dataframe` | `list[str]` | `dtype == pl.Datetime or dtype == pl.Date` |

### Null Imputation

**Source:** `preprocessing.py:23` (numeric), `preprocessing.py:31` (categorical)

| Node | Strategy | Details |
|------|----------|---------|
| `fill_nulls_numeric` | **Median** imputation | `pl.col(c).fill_null(pl.col(c).median())` — robust to outliers |
| `fill_nulls_categorical` | **Mode** imputation | `df[c].mode().first()` — most frequent value |

**`null_filled_df`** merges filled numeric, filled categorical, and other columns via horizontal concatenation.

### Date Decomposition

**Source:** `preprocessing.py:63`

Extracts calendar features from each date column, then drops the original:

| Feature | Expression | Example |
|---------|------------|---------|
| `{prefix}_year` | `pl.col(col).dt.year()` | 2024 |
| `{prefix}_month` | `pl.col(col).dt.month()` | 6 |
| `{prefix}_day` | `pl.col(col).dt.day()` | 15 |
| `{prefix}_day_of_week` | `pl.col(col).dt.weekday()` | 1 (Monday) |

The prefix is derived by stripping `_date` or `_dt` suffixes from the original column name.

### Categorical Encoding

**Source:** `preprocessing.py:82`

Uses **ordinal encoding** via Polars native operations:

```
pl.col(col).cast(pl.Categorical).to_physical()
```

This converts each categorical value to its integer representation. No one-hot encoding is applied — GBDT models handle ordinal-encoded categories natively (especially CatBoost with `cat_features`).

---

## Target Transformation

**Source:** `src/tabular_blueprint/data/feature_engine.py:49`

### Skewness Detection

**Function:** `detect_target_skewness(y)` → `float`

**Library:** `scipy.stats.skew`

**Formula:**

```
skewness = E[(X - μ)³] / σ³
```

### Transform Methods

**Function:** `transform_target(y, method, skewness_threshold)`

| Method | Library | Formula | Constraint |
|--------|---------|---------|------------|
| `"log1p"` | NumPy | `y' = log(1 + y)` | None (works with zeros) |
| `"yeo-johnson"` | `sklearn.preprocessing.PowerTransformer("yeo-johnson")` | `y' = ((y+1)^λ - 1) / λ` if y≥0, `((−y+1)^(2−λ) − 1) / (2−λ)` if y<0` | Any values |
| `"box-cox"` | `sklearn.preprocessing.PowerTransformer("box-cox")` | `y' = (y^λ − 1) / λ` for λ≠0, `log(y)` for λ=0` | **y > 0** strictly |
| `"auto"` | — | Picks Box-Cox if all positive, Yeo-Johnson otherwise; skips if \|skew\| ≤ threshold | Default strategy |
| `"none"` | — | No transformation | — |

**Auto Logic:**
1. Compute skewness of `y`
2. If `|skewness| ≤ 1.0` (threshold): skip transform
3. If all `y > 0`: use Box-Cox
4. Otherwise: use Yeo-Johnson

**Inverse Transform:** The `_TargetTransformer` class stores the fitted scaler for exact inversion (`inverse_transform`), or uses `np.expm1` for log1p.

---

## Data Quality Audit (Label Noise)

**Source:** `src/tabular_blueprint/data/quality.py`

### `audit_data_quality(df, target_col)`

**Library:** `cleanlab`

**Method:**
1. Fits `LogisticRegression(max_iter=1000)` with 3-fold `cross_val_predict` to get out-of-sample `predict_proba`
2. Computes `get_label_quality_scores(y, pred_probs)` — per-sample quality score
3. Identifies `find_label_issues(y, pred_probs)` — flagged label noise indices ranked by self-confidence

**Output:**
| Field | Description |
|-------|-------------|
| `n_issues` | Number of flagged label issues |
| `noise_rate` | `n_issues / n_rows` |
| `flagged_indices` | Top 100 noisy row indices |
| `quality_scores` | Per-sample quality score |
| `mean_quality_score` | Average quality across all samples |

### `clean_noise(df, report, target_col, quality_threshold=0.5)`

Drops rows where `quality_score < threshold`. Falls back to dropping `flagged_indices` if quality scores are unavailable.

---

## Leakage Detection

**Source:** `src/tabular_blueprint/data/leakage.py:16`

**Function:** `detect_leakage(X, y, task, threshold, cv_folds)`

**Method:** Per-feature permutation importance on a naive linear baseline:

1. Compute baseline CV score with all features (LogisticRegression or Ridge)
2. For each feature column: shuffle it, re-evaluate CV score
3. `score_drop = baseline_score - permuted_score`
4. Flag feature if `score_drop > threshold`

**Interpretation:** A large drop when permuting a feature means the model relied heavily on it — potential data leakage (e.g., target-encoded features, future data).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.15 | Minimum score drop to flag a feature |
| `cv_folds` | 3 | Number of CV folds |

---

## Data Format Conversion

**Source:** `src/tabular_blueprint/data/adapter.py:9`

**Class:** `DataAdapter`

Converts Polars DataFrames to the format required by each model type:

| Format | Target Model | Conversion |
|--------|-------------|------------|
| `"numpy"` | GBDTs (CatBoost, LightGBM, XGBoost), TabPFN | `X.to_numpy()`, `y.to_numpy()` |
| `"tensor"` | PyTorch models (FT-Transformer) | `df.to_torch()` — zero-copy when possible; features → `torch.float32`, target → `float32` (regression) or `long` (classification) |
| `"dataset"` | HuggingFace Transformers | Tries `Dataset.from_polars` → `from_arrow` → `from_pyarrow` → `from_dict` for compatibility |
