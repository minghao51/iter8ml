import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Methodology & Benchmarks

    Comprehensive overview of all modeling methods, evaluation techniques,
    and the benchmarking system powering **tabular-blueprint**.

    ## Contents
    1. **Preprocessing** — null imputation, date decomposition, encoding, target transforms
    2. **Model Families** — GBDTs, Deep Learning, TabPFN, Baselines
    3. **Evaluation** — CV strategies, metrics, lift, calibration
    4. **HPO** — Optuna pruners, warmstarting, search spaces, parameter importance
    5. **Feature Engineering** — permutation importance, interactions, pruning
    6. **Drift Detection** — KS/Chi², PSI, Domain Classifier
    7. **Explainability** — SHAP TreeExplainer vs KernelExplainer
    8. **Pipeline Architecture** — Hamilton DAG orchestration
    9. **Benchmarks** — running and interpreting performance benchmarks
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Preprocessing Methodology
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Pipeline**: `raw_df → null fill → date decomposition → encoding → processed_df`

    | Step | Method | Details |
    |------|--------|---------|
    | Numeric nulls | **Median** imputation | `fill_null(col.median())` — robust to outliers |
    | Categorical nulls | **Mode** imputation | `[col].mode().first()` — most frequent value |
    | Date cols | **Decomposition** | `year`, `month`, `day`, `day_of_week` extracted; original dropped |
    | Encoding | **Ordinal** | `cast(Categorical).to_physical()` — no one-hot (GBDTs handle this natively) |

    **Target Transformation** (for skewed regression targets):
    - `log1p`, `yeo-johnson`, `box-cox`, or `auto` (detect skewness first)
    - Skewness formula: $$ \text{skew} = \frac{E[(X-\mu)^3]}{\sigma^3} $$
    - Applied only when $$|\text{skew}| > \text{threshold}$$ (default 1.0)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Model Families — Pros, Cons & Stats
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Gradient Boosted Decision Trees

    **Common principle**: Sequential ensemble where each tree fits the gradient residual of the prior ensemble:
    $$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

    #### CatBoost
    - **Algorithm**: Ordered boosting with symmetric trees
    - **Key advantage**: Native categorical support (ordered target statistics — no preprocessing needed)
    - **Training**: GPU/CPU via `task_type`
    - **HPO range**: `depth=(4,10)`, `lr=(0.01,0.2)`, `l2_leaf_reg=(1,10)`
    - **Pros**: Robust defaults, best categorical handling, good with small data
    - **Cons**: Slower training than LightGBM, higher memory on large datasets

    #### LightGBM
    - **Algorithm**: Leaf-wise (best-first) growth + GOSS/EFB
    - **Key advantage**: Fastest training, lowest memory
    - **Trick**: GOSS keeps high-gradient instances, EFB bundles sparse features
    - **HPO range**: `num_leaves=(15,127)`, `lr=(0.01,0.2)`, `subsample=(0.5,1.0)`
    - **Pros**: Lightning fast, scales to millions of rows, low memory
    - **Cons**: Can overfit with deep leaf-wise trees, needs careful `num_leaves` tuning

    #### XGBoost
    - **Algorithm**: Level-wise growth with regularized objective
    - **Regularization**: $$\Omega(f) = \gamma T + \frac{1}{2}\lambda\|w\|^2$$
    - **Key advantage**: Mature ecosystem, best regularization controls
    - **HPO range**: `max_depth=(3,12)`, `lr=(0.01,0.2)`, `gamma=(0,5)`
    - **Pros**: Best regularization (gamma, lambda), battle-tested, extensive community
    - **Cons**: Slower than LightGBM, memory-heavy with deep trees

    | Aspect | CatBoost | LightGBM | XGBoost |
    |--------|----------|----------|---------|
    | Tree growth | Symmetric | Leaf-wise | Level-wise |
    | Categoricals | Native (ordered TS) | Ordinal encode | Ordinal encode |
    | Speed | Medium | Fastest | Medium-slow |
    | Memory | Medium | Lowest | Medium-high |
    | Small data | ★★★ | ★★ | ★★ |
    | Large data | ★★ | ★★★ | ★★ |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Deep Learning Models

    #### FT-Transformer
    - **Architecture**: Feature Tokenizer → Transformer Encoder → Prediction Head
    - **Flow**: Embed each feature → add sequence dim → TransformerEncoder → LayerNorm → ReLU → Dropout → Linear
    - **Attention**: $$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
    - **Requirements**: GPU with >12 GB VRAM recommended
    - **HParams**: `lr=1e-4`, `d_hidden=128`, `n_heads=4`, `n_layers=3`
    - **Pros**: Captures feature interactions via attention, SOTA for tabular DL
    - **Cons**: Needs GPU, slow training, overfits small data (<10K rows)

    #### TabNet
    - **Architecture**: Sequential attention masks for instance-wise feature selection
    - **Mechanism**: $$M[i] = \text{sparsemax}(f_e(a[i-1]))$$ — attention decides which features matter per step
    - **Requirements**: GPU with >8 GB VRAM
    - **HParams**: `lr=1e-3`, `batch_size=256`, `n_epochs=50`
    - **Pros**: Interpretable (instance-wise feature masks), self-regularizing
    - **Cons**: Slower than GBDTs, less performant on clean tabular data

    #### TabPFN v2
    - **Paradigm**: Prior-data fitted network — no gradient training at inference
    - **Mechanism**: $$\hat{y} = f_\theta(X_{\text{train}}, y_{\text{train}}, x_{\text{test}})$$ — in-context learning
    - **Library**: `tabpfn` (frozen transformer, pretrained on millions of synthetic datasets)
    - **Constraints**: Max 50,000 rows, GPU recommended
    - **Pros**: Zero-shot (no hyperparameter tuning), strong on small data
    - **Cons**: Max 50K rows, requires GPU for reasonable speed, limited to smaller feature sets

    ### Baselines
    | Model | Method | When to look |
    |-------|--------|-------------|
    | NaiveBaseline | Predicts mean/mode constant | Floor — any model must beat this |
    | LinearBaseline | LogisticRegression / Ridge | Simple signal — large lift here means nonlinear patterns exist |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model Selection Logic (ModelSelector)

    Hardware-aware routing based on dataset size and available GPU:

    | Condition | Models selected |
    |-----------|----------------|
    | Always | `catboost`, `lightgbm`, `xgboost` (GBDTs) |
    | `n_rows < 500K` | All GBDTs |
    | `n_rows >= 500K` | `lightgbm`, `xgboost` (skip CatBoost) |
    | GPU + `vram > 12GB` + `n_rows >= 50K` | `ft_transformer` |
    | GPU + `vram > 8GB` | `tabnet` |
    | Always | `naive_baseline`, `linear_baseline` |
    | GPU present | `tabpfn` |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Evaluation Methodology
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cross-Validation

    | Strategy | Splitter | Use case |
    |----------|----------|----------|
    | `kfold` | `KFold(shuffle=True)` | General purpose |
    | `stratified` | `StratifiedKFold` | Imbalanced classification |
    | `timeseries` | `TimeSeriesSplit` | Temporal data |

    **Evaluator**: Fresh model per fold → fit → predict → score → `np.mean()` aggregation.

    ### Metrics

    **Classification:**
    - **ROC AUC**: $$AUC = P(\text{score(positive)} > \text{score(negative)})$$ — ranking quality, threshold-independent
    - **F1 Macro**: $$F1 = \frac{1}{C}\sum_c 2\frac{P_c\cdot R_c}{P_c+R_c}$$ — balanced per-class harmonic mean
    - **Accuracy**: $$\text{Acc} = \frac{TP+TN}{N}$$ — overall correctness
    - **Log Loss**: $$-\frac{1}{N}\sum\sum y_{ic}\log(p_{ic})$$ — probability calibration quality

    **Regression:**
    - **RMSE**: $$\sqrt{\frac{1}{N}\sum(y_i-\hat{y}_i)^2}$$ — penalizes large errors quadratically
    - **MAE**: $$\frac{1}{N}\sum|y_i-\hat{y}_i|$$ — average absolute error
    - **R²**: $$1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$$ — variance explained

    ### Lift

    $$\text{lift} = \frac{\text{model} - \text{baseline}}{|\text{baseline}|}$$

    Sign flipped for lower-is-better metrics (`rmse`, `mae`, `log_loss`).

    ### Calibration

    | Method | Formula | Property |
    |--------|---------|----------|
    | **Platt scaling** | $$P(y=1|x) = 1 / (1 + \exp(A\cdot f(x) + B))$$ | Parametric, works with less data |
    | **Isotonic** | $$P(y=1|x) = \text{isotonic}(f(x))$$ | Non-parametric, more flexible, needs more data |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Hyperparameter Optimization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pruners

    | Pruner | Behavior | Best for |
    |--------|----------|----------|
    | **MedianPruner** | Stops trials below historical median | Default — safe, cheap |
    | **HyperbandPruner** | Adaptive resource allocation | Many trials, expensive evaluations |
    | **NopPruner** | All trials run to completion | Small studies (<20 trials) |

    ### Search Space Sampling

    Parameters defined as `(low, high, scale)` tuples:
    - `(1, 10)` → `suggest_int` (both int)
    - `(0.01, 0.2)` → `suggest_float` (any float)
    - `(0.01, 0.2, "log")` → `suggest_float(log=True)`

    ### Per-Model Search Spaces

    | Model | Key params (range, scale) |
    |-------|--------------------------|
    | CatBoost | `depth` (4-10), `lr` (0.01-0.2, log), `l2_leaf_reg` (1-10, log) |
    | LightGBM | `num_leaves` (15-127), `lr` (0.01-0.2, log), `subsample` (0.5-1.0) |
    | XGBoost | `max_depth` (3-12), `lr` (0.01-0.2, log), `gamma` (0-5) |
    | FT-Transformer | `lr` (1e-5-1e-3, log), `d_hidden` (64-256), `n_layers` (2-6) |
    | TabNet | `lr` (1e-4-1e-2, log), `batch_size` (64-512) |

    ### Warmstarting

    Pre-populates a new study from historical JSONL events. Distribution inferred from parameter names:
    - `*depth*`, `*n_estimators*` → `IntDistribution`
    - `*lr*`, `*learning_rate*` → `FloatDistribution(log=True)`
    - Boolean values → `CategoricalDistribution([True, False])`

    ### Parameter Importance (PedAnova)

    Permutation-based ANOVA that ranks hyperparameters by impact on the objective.
    More robust than frequency-based methods with small trial counts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Automated Feature Engineering
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pipeline

    1. **Top-K selection**: Permutation importance on a fitted model, keep top K
       $$ \text{importance}_j = \frac{1}{K}\sum_k \big[\text{score}(X) - \text{score}(X_{\text{perm},j,k})\big] $$

    2. **Interaction discovery**: Test pairwise `multiply` / `ratio` features via CV lift
       - If lift > threshold → add interaction as new feature

    3. **Pruning** (optional): Drop features below `min_importance`

    | Parameter | Default | Range |
    |-----------|---------|-------|
    | `afe_top_k` | 10 | 5–50 |
    | `afe_lift_threshold` | 0.01 | 0.001–0.1 |
    | `afe_prune_min_importance` | 0.001 | 0.0001–0.01 |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Drift Detection Methods
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### KS / Chi-squared (Univariate)

    - **Numeric**: Kolmogorov-Smirnov test — $$D = \max|F_{\text{ref}}(x) - F_{\text{live}}(x)|$$
    - **Categorical**: Chi-squared — $$\chi^2 = \sum\frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$
    - **H₀**: Distributions are identical. Drift if `p < α` (default 0.05)

    **Pros**: Fast, interpretable per-column. **Cons**: Univariate only, misses multivariate shifts.

    ### PSI (Population Stability Index)

    $$\text{PSI} = \sum_i (p_i^{\text{live}} - p_i^{\text{ref}}) \cdot \ln\left(\frac{p_i^{\text{live}}}{p_i^{\text{ref}}}\right)$$

    where `i` indexes bins. Severity: `<0.1` none, `0.1–0.25` moderate, `>0.25` severe.

    **Pros**: Industry standard in finance, unified scale. **Cons**: Bin-count dependent, univariate.

    ### Domain Classifier (Multivariate)

    Trains a classifier to distinguish reference vs live data. AUC > 0.7 → drift detected.

    **Pros**: Detects multivariate drift, sensitive to complex shifts.
    **Cons**: Black box, less interpretable, needs enough data to train a classifier.

    | Method | Type | Sensitivity | Speed |
    |--------|------|-------------|-------|
    | KS/Chi² | Univariate | Per-column shifts | ★★★ |
    | PSI | Univariate | Distribution magnitude | ★★★ |
    | Domain Classifier | Multivariate | Complex interactions | ★★ |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Explainability (SHAP)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Shapley value** for feature `i`:
    $$\phi_i = \sum_{S \subseteq F\setminus\{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} \cdot \big[f(S\cup\{i\}) - f(S)\big]$$

    | Model type | SHAP method | Complexity |
    |------------|-------------|------------|
    | LightGBM, XGBoost, CatBoost | **TreeExplainer** | O(TLD²) — exact |
    | All others | **KernelExplainer** | O(2^F · sample) — approximate |

    TreeExplainer is exact and fast — it exploits the tree structure to avoid exponential subset enumeration.
    KernelExplainer is model-agnostic but slow — it evaluates the model on a background sample of 100 rows with weighted linear regression on feature coalitions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Pipeline Architecture (Hamilton DAG)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The pipeline uses **Hamilton DAG** for declarative orchestration. Each function = a DAG node;
    parameters = dependencies; Hamilton resolves the execution order.

    ### 7 Training Modules
    ```
    preprocessing → data_preparation → model_selection → baselines
        → feature_engineering (conditional AFE) → model_training → state_generation
    ```

    ### 5 Pipeline Modes
    | Mode | Terminal node | Purpose |
    |------|--------------|---------|
    | `TRAINING` | `training_state` | Full experiment |
    | `DRIFT` | `drift_report` | Dataset comparison |
    | `EXPORT` | `processed_dataframe` | Champion export |
    | `HPO` | `processed_dataframe` | Hyperparameter search |
    | `INFERENCE` | `processed_dataframe` | Batch prediction |

    **Fallback**: If Hamilton unavailable, the imperative path runs (same logic, manual orchestration).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Benchmark System
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The benchmark framework measures **fit time**, **predict time**, and **predict_proba time**
    with warmup runs + measured runs for stable statistics.

    ### Protocol

    - **Warmup**: 2 runs (discarded, JIT/GC stabilization)
    - **Measured**: 5 runs (timed via `time.perf_counter`)
    - **Tracking**: mean, median, stdev, min, max
    - **Memory**: Optional peak memory delta via `psutil`

    ### Predefined Data Sizes

    ```python
    DATA_SIZES = [
        (500, 10),      # Small — baseline, quick iteration
        (5_000, 20),    # Medium — typical experiments
        (20_000, 50),   # Large — stress test
    ]
    ```

    ### Categories

    | Category | Benchmarks |
    |----------|------------|
    | `model_fit` | CatBoost, LightGBM, XGBoost fit time |
    | `model_predict` | Predict time at 1K / 10K samples |
    | `data_adapter` | Numpy & Tensor adapter conversion |
    | `feature_engineering` | Top-K, interactions, pruning, target transform |
    | `pipeline` | Full data preparation pipeline |
    | `data_io` | DataFrame hashing |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Run Benchmarks
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    root = Path.cwd().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from tabular_blueprint.models.factory import get_model_class
    from benchmarks.benchmark_utils import make_numpy
    from benchmarks.benchmark_models import bench_model_fit

    available = ["catboost", "lightgbm", "xgboost", "naive_baseline", "linear_baseline"]
    available_info = [(m, get_model_class(m).__name__) for m in available]
    f"Available models: {len(available)}"
    return bench_model_fit, make_numpy


@app.cell
def _(bench_model_fit, make_numpy, mo):
    @mo.persistent_cache
    def run_fit_benchmarks():
        sizes_for_demo = [(500, 10), (5000, 20)]
        results = []
        for n_samples, n_features in sizes_for_demo:
            X, y = make_numpy(n_samples, n_features)
            for model_name in ["catboost", "lightgbm", "xgboost"]:
                r = bench_model_fit(model_name, X, y, "classification", warmup=1, runs=3)
                results.append(r)
        return results

    results = run_fit_benchmarks()
    results
    return (results,)


@app.cell
def _(results):
    from benchmarks.benchmark_utils import _print_plain

    _print_plain(results)
    return


@app.cell
def _(mo, results):
    table_lines = []
    for br in sorted(results, key=lambda x: x.mean):
        param = ", ".join(f"{k}={v}" for k, v in br.params.items())
        table_lines.append(
            f"| {br.name:35s} | {param:25s} | {br.mean:.4f}s | {br.median:.4f}s | {br.stdev:.4f}s |"
        )
    table = (
        "| Benchmark | Params | Mean | Median | StdDev |\n"
        "|-----------|--------|------|--------|--------|\n" + "\n".join(table_lines)
    )
    mo.md(f"```\n{table}\n```")
    return


if __name__ == "__main__":
    app.run()
