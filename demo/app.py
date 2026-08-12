"""iter8ml live demo — upload a CSV, get a leaderboard + SHAP in the browser.

Gradio app for Hugging Face Spaces (Gradio SDK). The heavy lifting is the
testable :func:`run_analysis` core (no Gradio dependency); :func:`create_demo`
wraps it in a UI and is built lazily so the core can be imported and unit-run
without Gradio installed.

Deploy: see ``demo/README.md``. Local run::

    uv run --with gradio python demo/app.py
"""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering (HF Spaces / CI)

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from iter8ml import ExperimentConfig, ExperimentSession, TaskType, load_data
from iter8ml.config import HardwareProfile
from iter8ml.constants import CVStrategy
from iter8ml.data.adapter import DataAdapter
from iter8ml.engine.models.factory import get_model_class
from iter8ml.workspace import Workspace

# Cap OpenMP threads before any GBDT library loads libgomp. GBDT libs load lazily
# on the first get_model_class() call inside run_analysis(), so configuring at
# module-import time is early enough. On hybrid-core (P+E) CPUs under Linux/WSL2,
# libgomp deadlocks across all cores (Phase-1 issue 1.6b).
HardwareProfile.configure_omp_threads()

# ---------------------------------------------------------------------------
# Demo guard rails (free-tier-friendly)
# ---------------------------------------------------------------------------
MAX_ROWS = 20_000  # sample down above this to keep a run well under a minute
SHAP_SAMPLE = 500  # points plotted in the beeswarm
CV_FOLDS = 5
MODELS = ["catboost", "xgboost"]  # fast CPU GBDTs -> a real 2-way leaderboard

# Bundled sample, co-located with the app so the demo runs on the released PyPI
# package (no dependency on the in-package datasets module).
SAMPLE_PATH = str(Path(__file__).parent / "telco_churn.parquet")


def _ordinal_encode(X: np.ndarray) -> np.ndarray:
    """Ordinal-encode object columns so GBDTs/SHAP get numeric input.

    Mirrors ``benchmarks/openml_benchmark.py``.
    """
    from sklearn.preprocessing import OrdinalEncoder

    if not (X.dtype == object or (hasattr(X.dtype, "kind") and X.dtype.kind in ("U", "S", "O"))):
        return X
    for i in range(X.shape[1]):
        if X[:, i].dtype == object:
            with contextlib.suppress(ValueError, TypeError):
                X[:, i] = X[:, i].astype(float)
    str_cols = [i for i in range(X.shape[1]) if X[:, i].dtype == object]
    if str_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[:, str_cols] = enc.fit_transform(X[:, str_cols])
    return X.astype(np.float64)


def run_analysis(
    csv_path: str, target_col: str, task: str
) -> tuple[pl.DataFrame, plt.Figure | None, str]:
    """Run the iter8ml loop on a CSV. Returns ``(leaderboard, shap_fig, summary)``.

    Isolated per call via a throwaway workspace; safe to invoke concurrently.
    Raises ``ValueError`` on bad input; callers should catch for friendly UX.
    """
    df = load_data(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {df.columns}")

    sampled = ""
    if len(df) > MAX_ROWS:
        orig = len(df)
        df = df.sample(n=MAX_ROWS, seed=42)
        sampled = f" (sampled to {MAX_ROWS:,} from {orig:,})"

    task_enum = TaskType.CLASSIFICATION if task == "classification" else TaskType.REGRESSION
    metrics = ["roc_auc", "f1_macro"] if task == "classification" else ["rmse", "r2"]

    # Per-request isolated workspace (no cross-request collision, no persisted data).
    with tempfile.TemporaryDirectory() as tmp:
        ws = Workspace(root=Path(tmp))
        ws.init()
        session = ExperimentSession(workspace=ws, tracker=None)
        config = ExperimentConfig(
            name="demo",
            task=task_enum,
            target_col=target_col,
            data_path="",
            models=MODELS,
            cv_folds=CV_FOLDS,
            cv_strategy=CVStrategy.STRATIFIED if task == "classification" else CVStrategy.KFOLD,
            metrics=metrics,
            shap_enabled=False,
            random_seed=42,
            max_workers=1,
        )
        session.run(config, df)
        leaderboard = session.leaderboard()

    # SHAP beeswarm on the champion (CatBoost), refit on an encoded matrix.
    adapter = DataAdapter()
    feat_names = [c for c in df.columns if c != target_col]
    X, y = adapter.transform(df, target_col)
    X = _ordinal_encode(X)
    if task == "classification" and (y.dtype == object or y.dtype.kind in ("U", "S", "O")):
        from sklearn.preprocessing import LabelEncoder

        y = LabelEncoder().fit_transform(np.asarray(y).astype(str))

    # Pick the leaderboard champion (excluding naive/linear baselines) for SHAP.
    lb_main = leaderboard.filter(~pl.col("model").str.contains("baseline"))
    if lb_main.height:
        champ_row = lb_main.row(0, named=True)
        champ_name = str(champ_row["model"])
        champ_key = champ_name.lower()
        champ_line = (
            f"Champion: {champ_name} {champ_row['primary_metric']}={champ_row['primary_score']:.4f}"
        )
    else:
        champ_name = "CatBoost"
        champ_key = "catboost"
        champ_line = "Champion: n/a"

    try:
        champion = get_model_class(champ_key)(task=task)
    except Exception:  # fall back if the key isn't a registered model
        champion = get_model_class("catboost")(task=task)
    champion.fit(X, y)

    import shap

    sv = shap.TreeExplainer(champion._model)(X[:SHAP_SAMPLE])
    fig, _ = plt.subplots(figsize=(8.5, 5.5))
    shap.plots.beeswarm(sv, max_display=12, show=False)
    plt.title(f"SHAP — what drives '{target_col}' ({champ_name})")
    plt.tight_layout()

    summary = (
        f"{len(df):,} rows x {df.width - 1} features{sampled} | task={task}\n"
        f"{champ_line}\n"
        f"Top SHAP driver: {feat_names[int(np.argsort(np.abs(sv.values).mean(0))[-1])]}"
    )
    return leaderboard, fig, summary


# ---------------------------------------------------------------------------
# Gradio UI (built lazily so the core above stays importable without gradio)
# ---------------------------------------------------------------------------
def create_demo():  # pragma: no cover (UI shell)
    import gradio as gr

    def _on_upload(file):
        if file is None:
            return gr.update(choices=[], value=None)
        try:
            cols = load_data(file).columns
        except Exception as e:
            return gr.update(choices=[], value=None, label=f"Target (load error: {e})")
        guess = cols[-1]  # last column is the target in many exported datasets
        return gr.update(choices=list(cols), value=guess)

    def _on_sample():
        cols = load_data(SAMPLE_PATH).columns
        return SAMPLE_PATH, gr.update(choices=list(cols), value="Churn"), "classification"

    def _run(file, target, task):
        if not file or not target:
            return pl.DataFrame(), None, "Upload a CSV or click 'Use sample', then Run."
        try:
            return run_analysis(file, target, task)
        except Exception as e:
            import traceback

            return pl.DataFrame(), None, f"Error: {e}\n\n{traceback.format_exc()}"

    with gr.Blocks(title="iter8ml demo", theme=gr.themes.Soft()) as ui:
        gr.Markdown(
            "# iter8ml — tabular ML in one click\n"
            "Upload a CSV (or use the bundled Telco Churn sample), pick the target "
            "column, and get a cross-validated leaderboard + SHAP explanation. "
            f"Capped at {MAX_ROWS:,} rows, CatBoost-only, for speed."
        )
        with gr.Row():
            csv_in = gr.File(label="CSV file", file_types=[".csv"])
            sample_btn = gr.Button("Use Telco Churn sample")
        with gr.Row():
            target = gr.Dropdown(label="Target column", choices=["Churn"], value="Churn")
            task = gr.Radio(["classification", "regression"], value="classification", label="Task")
            run_btn = gr.Button("Run", variant="primary")
        summary = gr.Textbox(label="Summary", lines=3)
        with gr.Row():
            leaderboard = gr.Dataframe(label="Leaderboard (5-fold CV)")
            shap_plot = gr.Plot(label="SHAP — champion")

        csv_in.change(_on_upload, csv_in, target)
        sample_btn.click(_on_sample, None, [csv_in, target, task])
        run_btn.click(_run, [csv_in, target, task], [leaderboard, shap_plot, summary])

    return ui


# HF Spaces convention: a module-level `demo` object.
try:
    demo = create_demo()
except ImportError:
    demo = None  # Gradio not installed (e.g. importing the core for tests)


if __name__ == "__main__" and demo is not None:  # pragma: no cover
    demo.launch()
