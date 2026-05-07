# Models

Reference for all model implementations, selection logic, and configuration.

---

## Baseline Models

### NaiveBaseline

**Source:** `src/tabular_blueprint/models/baselines.py:9`

**Description:** Predicts a single constant value for all samples — the mean for regression, the mode for classification. Used as a floor baseline to contextualize real model performance.

**Library:** NumPy only

**Behavior:**
- **Regression:** `pred = mean(y_train)`
- **Classification:** `pred = mode(y_train)`, `predict_proba` returns one-hot at the mode class

**Options:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `str` | `"classification"` | `"classification"` or `"regression"` |

---

### LinearBaseline

**Source:** `src/tabular_blueprint/models/baselines.py:63`

**Description:** Simple linear model as a stronger baseline. Uses scikit-learn's `LogisticRegression` for classification and `Ridge` regression for regression tasks.

**Library:** scikit-learn

**Options:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `str` | `"classification"` | Determines LogisticRegression vs Ridge |
| `max_iter` | `int` | `1000` | LogisticRegression max iterations |
| `alpha` | `float` | `1.0` | Ridge L2 regularization strength |

---

## Gradient Boosted Decision Trees (GBDTs)

All GBDT models extend `BaseGBDTModel` (`src/tabular_blueprint/models/gbdt_base.py`), which provides a common interface: `fit`, `predict`, `predict_proba`, `save`, `load`.

### CatBoost

**Source:** `src/tabular_blueprint/models/conventional/catboost_model.py:10`

**Description:** Yandex's CatBoost — handles categorical features natively, robust to overfitting with ordered boosting.

**Library:** `catboost`

**Mathematical Formulation:**
Ordered boosting builds decision trees sequentially where each tree fits the residual (gradient of the loss) of the ensemble so far:

```
F_m(x) = F_{m-1}(x) + η · h_m(x)
```

where `η` is the learning rate and `h_m` is the m-th decision tree trained on the gradient residual. CatBoost uses ordered target statistics for categorical encoding, avoiding target leakage.

**Default Hyperparameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `iterations` | 1000 | Number of boosting iterations |
| `depth` | 6 | Tree depth |
| `learning_rate` | 0.05 | Step size shrinkage |
| `l2_leaf_reg` | 3.0 | L2 regularization coefficient |
| `early_stopping_rounds` | 50 | Stop if no improvement for N rounds |
| `task_type` | `"CPU"` | `"CPU"` or `"GPU"` |
| `random_seed` | 42 | Reproducibility |

**HPO Search Space:**
| Parameter | Range | Scale |
|-----------|-------|-------|
| `depth` | (4, 10) | linear |
| `learning_rate` | (0.01, 0.2) | log |
| `l2_leaf_reg` | (1.0, 10.0) | log |
| `iterations` | (500, 3000) | linear |

---

### LightGBM

**Source:** `src/tabular_blueprint/models/conventional/lightgbm_model.py:9`

**Description:** Microsoft's LightGBM — uses leaf-wise tree growth (best-first) and histogram-based splitting for fast training on large datasets.

**Library:** `lightgbm` (native `lgb.train` API)

**Mathematical Formulation:**
LightGBM uses Gradient-based One-Side Sampling (GOSS) to keep large-gradient instances and randomly drop small-gradient ones, and Exclusive Feature Bundling (EFB) to reduce the number of features:

```
GOSS: keep instances where |grad_i| > threshold, sample fraction of the rest
EFB: bundle mutually exclusive features into a single feature
```

Uses `lgb.Dataset` for efficient in-memory representation and `lgb.train` for the training loop with `num_boost_round` controlled by `n_estimators`.

**Default Hyperparameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 1000 | Number of boosting rounds |
| `max_depth` | -1 | No limit (leaf-wise growth) |
| `learning_rate` | 0.05 | Shrinkage rate |
| `num_leaves` | 31 | Max leaves per tree |
| `min_child_samples` | 20 | Min data per leaf |
| `subsample` | 0.8 | Row subsampling ratio |
| `colsample_bytree` | 0.8 | Feature subsampling ratio |

**HPO Search Space:**
| Parameter | Range | Scale |
|-----------|-------|-------|
| `max_depth` | (3, 12) | linear |
| `learning_rate` | (0.01, 0.2) | log |
| `num_leaves` | (15, 127) | linear |
| `min_child_samples` | (5, 50) | linear |
| `subsample` | (0.5, 1.0) | linear |
| `colsample_bytree` | (0.5, 1.0) | linear |

---

### XGBoost

**Source:** `src/tabular_blueprint/models/conventional/xgboost_model.py:9`

**Description:** XGBoost — uses `hist` tree method for fast approximate splitting and the native `xgb.train` + `DMatrix` API.

**Library:** `xgboost` (native `xgb.train` API)

**Mathematical Formulation:**
XGBoost optimizes a regularized objective:

```
L(φ) = Σ l(y_i, ŷ_i) + Σ Ω(f_k)
Ω(f) = γT + ½λ||w||²
```

where `T` is the number of leaves, `w` are leaf weights, `γ` is `gamma` (minimum split loss), and `λ` is L2 regularization. The `hist` tree method uses histogram-based approximate split finding for O(n) complexity per feature.

**Default Hyperparameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 1000 | Number of boosting rounds |
| `max_depth` | 6 | Maximum tree depth |
| `learning_rate` | 0.05 | Step size |
| `subsample` | 0.8 | Row subsampling |
| `colsample_bytree` | 0.8 | Feature subsampling |
| `gamma` | 0.0 | Minimum split loss reduction |
| `tree_method` | `"hist"` | Histogram-based splitting |

**HPO Search Space:**
| Parameter | Range | Scale |
|-----------|-------|-------|
| `max_depth` | (3, 12) | linear |
| `learning_rate` | (0.01, 0.2) | log |
| `subsample` | (0.5, 1.0) | linear |
| `colsample_bytree` | (0.5, 1.0) | linear |
| `gamma` | (0.0, 5.0) | linear |

---

## Deep Learning Models

### FT-Transformer

**Source:** `src/tabular_blueprint/models/deep/ft_transformer.py:42`

**Description:** Feature Tokenizer Transformer — embeds all features into a shared latent space, then processes through a standard Transformer encoder. Requires GPU with >12 GB VRAM.

**Library:** PyTorch, HuggingFace `accelerate`

**Mathematical Formulation:**
1. Feature embedding: `e = W·x + b` where `W ∈ R^{d×n_features}`, projecting each sample to a `d_hidden`-dimensional vector
2. Transformer encoding: `z = TransformerEncoder(LayerNorm(e))` with multi-head self-attention:
   ```
   Attention(Q, K, V) = softmax(QK^T / √d_k) · V
   ```
   Each encoder layer includes: multi-head attention → residual + LayerNorm → FFN → residual + LayerNorm
3. Prediction head: `output = Linear(Dropout(ReLU(LayerNorm(z))))`

**Architecture (`_FTTransformer` module):**
```
Input → Linear(n_features, d_hidden)
      → unsqueeze(1)  # add sequence dim
      → TransformerEncoder(n_layers × TransformerEncoderLayer)
      → squeeze(1)
      → LayerNorm → ReLU → Dropout → Linear(d_hidden, n_classes)
```

**Default Hyperparameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_epochs` | 100 | Training epochs |
| `batch_size` | 128 | Mini-batch size |
| `learning_rate` | 1e-4 | AdamW learning rate |
| `n_heads` | 4 | Multi-head attention heads |
| `d_hidden` | 128 | Hidden dimension (embedding size) |
| `n_layers` | 3 | Transformer encoder layers |
| `dropout` | 0.1 | Dropout rate |

**HPO Search Space:**
| Parameter | Range | Scale |
|-----------|-------|-------|
| `learning_rate` | (1e-5, 1e-3) | log |
| `d_hidden` | (64, 256) | linear |
| `n_heads` | (2, 8) | linear |
| `n_layers` | (2, 6) | linear |
| `dropout` | (0.0, 0.3) | linear |

**Training Details:**
- Optimizer: AdamW
- Loss: `CrossEntropyLoss` (classification) / `MSELoss` (regression)
- Acceleration: HuggingFace `Accelerator` for mixed-precision GPU training

---

### TabNet

**Source:** `src/tabular_blueprint/models/deep/tabnet_model.py:18`

**Description:** TabNet via `pytorch-tabular` — uses sequential attention to select features at each decision step, combining the interpretability of tree-based models with deep learning.

**Library:** `pytorch-tabular`

**Mathematical Formulation:**
TabNet uses attentive feature transformation through a sequence of steps:
1. Feature selection via sparse attention masks: `M[i] = sparsemax(f_e(a[i-1]))`
2. Feature processing through shared and step-specific FC layers
3. Final output: aggregation of all step outputs

This enables instance-wise feature selection at each decision step.

**Default Hyperparameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_epochs` | 50 | Training epochs |
| `batch_size` | 256 | Mini-batch size |
| `learning_rate` | 1e-3 | Learning rate |
| `early_stopping` | `"valid_loss"` | Early stopping metric |
| `early_stopping_patience` | 10 | Patience epochs |

**HPO Search Space:**
| Parameter | Range | Scale |
|-----------|-------|-------|
| `learning_rate` | (1e-4, 1e-2) | log |
| `batch_size` | (64, 512) | linear |
| `n_epochs` | (20, 100) | linear |

---

### TextEncoder

**Source:** `src/tabular_blueprint/models/deep/text_encoder.py:9`

**Description:** DeBERTa-v3 text-to-embedding encoder. Converts text columns into dense vector features via CLS token pooling from a pre-trained transformer.

**Library:** HuggingFace `transformers`, PyTorch

**Behavior:**
1. Tokenizes text with `AutoTokenizer` (max_length=128, padding + truncation)
2. Forward pass through `AutoModel` (DeBERTa-v3-base)
3. Extracts `last_hidden_state[:, 0, :]` — the CLS token embedding
4. Appends embedding dimensions as new columns: `{col}_emb_{i}`

**Options:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `"microsoft/deberta-v3-base"` | HuggingFace model identifier |
| `device` | `str` | auto-detect | `"cuda"` or `"cpu"` |
| `max_length` | `int` | 128 | Max token length |

---

## Tabular Foundation Model

### TabPFN v2

**Source:** `src/tabular_blueprint/models/tabular_foundation/tabpfn_model.py:19`

**Description:** TabPFN v2 — a pre-trained transformer that performs in-context learning on tabular data. Treats the entire training set as context and predicts without gradient-based training.

**Library:** `tabpfn`

**Mathematical Formulation:**
TabPFN is a prior-fitted network: it was pre-trained on millions of synthetic datasets to approximate Bayesian inference. At prediction time, the entire training set is fed as context (similar to in-context learning in LLMs):

```
ŷ = f_θ(X_train, y_train, x_test)
```

where `f_θ` is a frozen transformer trained via prior data augmentation. No gradient updates occur at inference.

**Constraints:**
- Maximum `50,000` rows (configurable via `max_rows`)
- GPU recommended but falls back to CPU with a warning

**Default Hyperparameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 4 | Number of ensembles |
| `device` | auto-detect | `"cuda"` or `"cpu"` |
| `max_rows` | 50,000 | Row limit guardrail |

**HPO Search Space:** None (frozen model, no tunable hyperparameters beyond `n_estimators`)

---

## Model Selection

### ModelSelector

**Source:** `src/tabular_blueprint/models/selector.py:6`

**Description:** Hardware-aware and data-size-aware model routing. Selects an ordered list of models to train based on dataset size, GPU availability, and VRAM.

**Routing Logic:**

| Condition | Models Selected |
|-----------|----------------|
| GPU present | + `tabpfn` |
| `n_rows < 500k` | `catboost`, `lightgbm`, `xgboost` |
| `n_rows >= 500k` | `lightgbm`, `xgboost` |
| `vram > 12 GB` and `n_rows >= 50k` | + `ft_transformer` |
| `vram > 8 GB` | + `tabnet` |
| `include_baselines=True` (default) | + `naive_baseline`, `linear_baseline` |

---

## Model Factory & Registry

### `_MODEL_REGISTRY`

**Source:** `src/tabular_blueprint/models/factory.py:5`

Lazy-import registry mapping model names to `(module_path, class_name)`:

| Name | Module | Class |
|------|--------|-------|
| `catboost` | `models.conventional.catboost_model` | `CatBoostModel` |
| `lightgbm` | `models.conventional.lightgbm_model` | `LightGBMModel` |
| `xgboost` | `models.conventional.xgboost_model` | `XGBoostModel` |
| `tabpfn` | `models.tabular_foundation.tabpfn_model` | `TabPFNModel` |
| `ft_transformer` | `models.deep.ft_transformer` | `FTTransformerModel` |
| `tabnet` | `models.deep.tabnet_model` | `TabNetModel` |
| `naive_baseline` | `models.baselines` | `NaiveBaseline` |
| `linear_baseline` | `models.baselines` | `LinearBaseline` |

**Key Functions:**
- `get_model_class(name)` → resolves and caches the class (lazy import)
- `validate_model_name(name)` → raises `ValueError` if unknown
- `available_model_names()` → returns sorted list of valid names
