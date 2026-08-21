"""Render benchmark results into a chart + markdown table for the README.

Reproducible from the committed benchmark output::

    uv run python benchmarks/render_results.py

Reads ``benchmarks/results/summary.json`` (produced by run_openml_benchmark.py),
writes a grouped bar chart to ``docs/img/benchmark_results.png`` and prints a
markdown table (mean ± std across CV folds) to stdout.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "benchmarks" / "results" / "summary.json"
OUT_IMG = ROOT / "docs" / "img" / "benchmark_results.png"

MODEL_COLORS = {"catboost": "#FF6B35", "lightgbm": "#4C78A8", "xgboost": "#54A24B"}
MODEL_ORDER = ["catboost", "lightgbm", "xgboost"]


def headline_metric(task: str, n_classes: int) -> str:
    """roc_auc (binary clf), f1_macro (multiclass — roc_auc is unstable on
    imbalanced folds), r2 (regression)."""
    if task == "regression":
        return "r2"
    return "roc_auc" if (n_classes or 2) <= 2 else "f1_macro"


def fmt(mean: float | None, std: float | None, best: bool) -> str:
    if mean is None:
        return "—"
    base = f"{mean:.3f}"
    if std is not None:
        base += f" ±{std:.3f}"
    return f"**{base}**" if best else base


def main() -> None:
    rows = json.loads(SUMMARY.read_text())
    # dataset -> {task, n, n_classes, models: {model: {mean, std}}}
    table: dict[str, dict] = {}
    for r in rows:
        if "error" in r:
            continue
        ds = r["dataset"]
        table.setdefault(
            ds,
            {
                "task": r["task"],
                "n": r["n_rows"],
                "n_classes": r.get("n_classes", 0),
                "models": {},
            },
        )
        table[ds]["models"][r["model"]] = {
            "mean": r["cv_scores"],
            "std": r.get("cv_std", {}),
        }

    clf = [d for d, v in table.items() if v["task"] == "classification"]
    reg = [d for d, v in table.items() if v["task"] == "regression"]

    # --- chart: 2 panels, headline metric per dataset ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    panels = [
        (axes[0], clf, "Classification (roc_auc / f1_macro)", (0.0, 1.02)),
        (axes[1], reg, "Regression (R²)", None),
    ]
    for ax, datasets, title, ylim in panels:
        n_ds = len(datasets)
        width = 0.26
        x = range(n_ds)
        for i, model in enumerate(MODEL_ORDER):
            vals = []
            for d in datasets:
                v = table[d]
                m = headline_metric(v["task"], v["n_classes"])
                vals.append(v["models"].get(model, {"mean": {}})["mean"].get(m, float("nan")))
            offset = (i - 1) * width
            ax.bar([xi + offset for xi in x], vals, width, label=model, color=MODEL_COLORS[model])
        ax.set_xticks(list(x))
        ax.set_xticklabels(datasets, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel("score")
        ax.axhline(0, color="0.7", linewidth=0.8)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle("iter8ml benchmark — 5-fold CV, default hyperparameters, CPU", y=1.02, fontsize=11)
    fig.tight_layout()
    OUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_IMG, dpi=150, bbox_inches="tight")
    print(f"chart -> {OUT_IMG.relative_to(ROOT)}")

    # --- markdown table: mean ± std, headline metric per dataset ---
    print("\n| Dataset | Task | N | Metric | CatBoost | LightGBM | XGBoost |")
    print("|---|---|--:|---|--:|--:|--:|")
    for ds in clf + reg:
        v = table[ds]
        m = headline_metric(v["task"], v["n_classes"])
        means = {mod: d["mean"].get(m) for mod, d in v["models"].items()}
        present = [x for x in means.values() if x is not None]
        best = max(present) if present else None
        cells = []
        for mod in MODEL_ORDER:
            mn = means[mod]
            sd = None
            if mn is not None:
                sd = v["models"].get(mod, {}).get("std", {}).get(m)
            is_best = best is not None and mn is not None and abs(mn - best) < 1e-9
            cells.append(fmt(mn, sd, is_best))
        print(f"| {ds} | {v['task']} | {v['n']:,} | {m} | {cells[0]} | {cells[1]} | {cells[2]} |")
    print(
        "\n_5-fold CV (mean ± std) · default hyperparameters · CPU · "
        "roc_auc (binary) / f1_macro (multiclass) / R² (regression) · best per row in bold._"
    )


if __name__ == "__main__":
    main()
