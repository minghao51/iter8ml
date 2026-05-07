# Drift Detection

Reference for the three drift detection methods: univariate statistical tests (KS/Chi-squared), Population Stability Index (PSI), and multivariate domain classifier.

---

## 1. Univariate Statistical Tests (KS / Chi-squared)

**Source:** `src/tabular_blueprint/monitoring/drift.py:22`

**Class:** `DriftDetector`

### Method

Compares each column of a reference DataFrame (training data) against a new DataFrame (production data) independently.

### Kolmogorov-Smirnov Test (Numeric Columns)

**Library:** `scipy.stats.ks_2samp`

**Mathematical Formulation:**

The KS test measures the maximum absolute difference between two empirical cumulative distribution functions:

```
D = max|F_ref(x) - F_live(x)|
```

**Hypothesis:**
- H₀: The reference and live distributions are identical
- H₁: The distributions differ

Drift is detected when `p_value < α` (default α = 0.05).

### Chi-Squared Test (Categorical Columns)

**Library:** `scipy.stats.chi2_contingency`

**Mathematical Formulation:**

Constructs a contingency table of category frequencies for reference vs live data:

```
χ² = Σ (O_ij - E_ij)² / E_ij
```

where `O_ij` is the observed count and `E_ij` is the expected count under the null hypothesis of independence.

**Hypothesis:**
- H₀: Category distributions are the same in reference and live
- H₁: Distributions differ

### Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.05 | Significance level for drift flagging |

### Output

`DriftReport` with per-column `ColumnDriftResult`:
- `column` — column name
- `p_value` — test p-value
- `drift_detected` — `p_value < alpha`
- `test_used` — `"ks_test"` or `"chi2_test"`

Global `drift_detected` is `True` if any column shows drift.

---

## 2. Population Stability Index (PSI)

**Source:** `src/tabular_blueprint/monitoring/psi_drift.py:28`

**Class:** `PSIDriftDetector`

### Mathematical Formulation

PSI quantifies the shift in a feature's distribution between reference and live data using binned proportions:

```
PSI = Σ_i (p_live_i - p_ref_i) × ln(p_live_i / p_ref_i)
```

where:
- `p_ref_i` = proportion of reference values in bin `i`
- `p_live_i` = proportion of live values in bin `i`
- Bins are computed via `np.percentile` on the combined distribution (10 bins default)
- Proportions are clipped to `[1e-6, ∞)` to avoid division by zero

### Thresholds

| PSI Value | Drift Level | Action |
|-----------|-------------|--------|
| ≤ 0.20 | `"none"` | No significant drift |
| 0.20 – 0.30 | `"moderate"` | Monitor closely |
| > 0.30 | `"severe"` | Investigation required |

### Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_bins` | 10 | Number of quantile bins |

### Output

`PSIDriftReport` with per-feature `FeaturePSI`:
- `feature` — column name
- `psi_value` — computed PSI
- `drift_level` — `"none"`, `"moderate"`, or `"severe"`

Only numeric columns are tested.

---

## 3. Domain Classifier (Multivariate)

**Source:** `src/tabular_blueprint/monitoring/domain_classifier.py:23`

**Class:** `DomainClassifierDriftDetector`

### Method

Detects multivariate drift by training a classifier to distinguish reference from live data:

1. Label reference rows as `0`, live rows as `1`
2. Stack into a single matrix `X`
3. Train `LogisticRegression` with K-fold CV
4. Evaluate via ROC AUC

### Mathematical Formulation

```
AUC = P(classifier(live_sample) > classifier(ref_sample))
```

**Interpretation:**
- `AUC ≈ 0.5` → reference and live are indistinguishable (no drift)
- `AUC > 0.7` → the classifier can tell them apart → drift detected

The default threshold of `0.7` balances sensitivity and false positive rate.

### Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.7 | AUC threshold for drift detection |
| `random_seed` | 42 | Reproducibility |
| `n_folds` | `min(5, max(2, n_min))` | Adaptive fold count based on smallest class |

### Output

`DomainDriftReport`:
- `drift_detected` — `auc_score > threshold`
- `auc_score` — mean cross-validated AUC
- `n_reference` / `n_live` — sample sizes

Only numeric columns are used.

---

## Drift Pipeline Integration

**Source:** `src/tabular_blueprint/pipelines/nodes/drift_detection.py:22`

Available as Hamilton DAG config variants:

| Variant | Method |
|---------|--------|
| `drift_report__psi` | PSI drift detection |
| `drift_report__domain` | Domain classifier drift |
| `drift_report__both` | Runs both PSI and domain classifier, merges reports |

**Orchestrator:** The drift detection pipeline is available via `PipelineExecutor.run_drift()` (`src/tabular_blueprint/pipelines/executor.py:208`) or directly via the monitoring classes (`DriftDetector`, `PSIDriftDetector`, `DomainClassifierDriftDetector`).
