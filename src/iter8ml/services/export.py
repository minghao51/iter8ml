"""Export champion models as portable prediction packages."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from iter8ml.services.registry import RegistryService

if TYPE_CHECKING:
    from iter8ml.workspace import Workspace

PREDICTOR_TEMPLATE = '''\
"""Auto-generated predictor for {model_name}.

Usage:
    from predictor import Predictor
    p = Predictor()
    predictions = p.predict("data.csv")
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

_HERE = Path(__file__).parent


def _build_preprocessing_driver():
    from hamilton import driver as h_driver
    from pipelines import preprocessing

    return h_driver.Builder().with_modules(preprocessing).build()


class Predictor:
    def __init__(self):
        with open(_HERE / "metadata.json") as f:
            self.meta = json.load(f)

        model_cls = self._load_model_class()
        self.model = model_cls(task=self.meta["task"])
        self.model.load(str(_HERE / "model.artifact"))

        try:
            self._dr = _build_preprocessing_driver()
        except ImportError:
            self._dr = None

    def _load_model_class(self):
        module_path, class_name = self.meta["model_class"]
        allowlisted = set()
        for item in self.meta.get("allowlisted_model_classes", []):
            if isinstance(item, list) and len(item) == 2:
                allowlisted.add((item[0], item[1]))
        requested = (module_path, class_name)
        if requested not in allowlisted:
            raise ValueError(
                "Model class in metadata is not allowlisted. "
                "This export may be tampered or incompatible."
            )
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        if self._dr is not None:
            result = self._dr.execute(["processed_dataframe"], inputs={{"df": df}})
            return result["processed_dataframe"]

        from pipelines.preprocessing import (
            fill_nulls_numeric,
            fill_nulls_categorical,
            decomposed_dates_df,
            encoded_df,
        )
        numeric_cols = df.select(pl.col(pl.Int64, pl.Float64, pl.Int32, pl.Float32)).columns
        cat_cols = df.select(pl.col(pl.Utf8, pl.Categorical)).columns
        date_cols = [
            c for c, dtype in df.schema.items()
            if dtype in (pl.Datetime, pl.Date)
        ]
        df = fill_nulls_numeric(df, numeric_cols)
        df = fill_nulls_categorical(df, cat_cols)
        df = decomposed_dates_df(df, date_cols)
        df = encoded_df(df, cat_cols)
        return df

    def predict(self, data_path: str) -> np.ndarray:
        df = pl.read_csv(data_path) if data_path.endswith(".csv") else pl.read_parquet(data_path)
        target_col = self.meta.get("target_col")
        if target_col and target_col in df.columns:
            df = df.drop(target_col)
        df = self._preprocess(df)
        X = df.to_numpy()
        return self.model.predict(X)

    def predict_proba(self, data_path: str) -> np.ndarray | None:
        df = pl.read_csv(data_path) if data_path.endswith(".csv") else pl.read_parquet(data_path)
        target_col = self.meta.get("target_col")
        if target_col and target_col in df.columns:
            df = df.drop(target_col)
        df = self._preprocess(df)
        X = df.to_numpy()
        return self.model.predict_proba(X)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to CSV or Parquet file")
    parser.add_argument("--proba", action="store_true", help="Return probabilities")
    args = parser.parse_args()

    p = Predictor()
    if args.proba:
        result = p.predict_proba(args.data_path)
    else:
        result = p.predict(args.data_path)

    if result is not None:
        print(result)
    else:
        print("Probabilities not available for this model.")
'''

PREPROCESSING_TEMPLATE = '''\
"""Standalone preprocessing helpers for exported predictors."""

from __future__ import annotations

import polars as pl
from polars import selectors as cs


def raw_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    return df


def numeric_columns(raw_dataframe: pl.DataFrame) -> list[str]:
    return raw_dataframe.select(cs.numeric()).columns


def categorical_columns(raw_dataframe: pl.DataFrame) -> list[str]:
    return raw_dataframe.select(cs.categorical() | cs.string()).columns


def date_columns(raw_dataframe: pl.DataFrame) -> list[str]:
    return [c for c, dtype in raw_dataframe.schema.items() if dtype in (pl.Datetime, pl.Date)]


def fill_nulls_numeric(
    raw_dataframe: pl.DataFrame,
    numeric_columns: list[str],
) -> pl.DataFrame:
    exprs = [pl.col(c).fill_null(pl.col(c).median()) for c in numeric_columns]
    return raw_dataframe.with_columns(exprs) if exprs else raw_dataframe


def fill_nulls_categorical(
    fill_nulls_numeric: pl.DataFrame,
    categorical_columns: list[str],
) -> pl.DataFrame:
    exprs = []
    for c in categorical_columns:
        mode_val = fill_nulls_numeric[c].mode().first()
        if mode_val is not None:
            exprs.append(pl.col(c).fill_null(mode_val))
    return fill_nulls_numeric.with_columns(exprs) if exprs else fill_nulls_numeric


def null_filled_df(
    fill_nulls_numeric: pl.DataFrame,
    fill_nulls_categorical: pl.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pl.DataFrame:
    numeric_part = fill_nulls_numeric.select([pl.col(c) for c in numeric_columns])
    cat_part = fill_nulls_categorical.select([pl.col(c) for c in categorical_columns])
    other_cols = [
        c
        for c in fill_nulls_numeric.columns
        if c not in numeric_columns and c not in categorical_columns
    ]
    other_part = fill_nulls_numeric.select(other_cols) if other_cols else pl.DataFrame()
    parts = [p for p in [other_part, numeric_part, cat_part] if p.width > 0]
    return pl.concat(parts, how="horizontal") if parts else fill_nulls_numeric


def decomposed_dates_df(
    null_filled_df: pl.DataFrame,
    date_columns: list[str],
) -> pl.DataFrame:
    exprs = []
    for col in date_columns:
        prefix = col.replace("_date", "").replace("_dt", "")
        exprs.extend(
            [
                pl.col(col).dt.year().alias(f"{prefix}_year"),
                pl.col(col).dt.month().alias(f"{prefix}_month"),
                pl.col(col).dt.day().alias(f"{prefix}_day"),
                pl.col(col).dt.weekday().alias(f"{prefix}_day_of_week"),
            ]
        )
    result = null_filled_df.with_columns(exprs) if exprs else null_filled_df
    return result.drop(date_columns) if date_columns else result


def encoded_df(
    decomposed_dates_df: pl.DataFrame,
    categorical_columns: list[str],
) -> pl.DataFrame:
    cat_cols = [c for c in categorical_columns if c in decomposed_dates_df.columns]
    if not cat_cols:
        return decomposed_dates_df
    exprs = [pl.col(col).cast(pl.Categorical).to_physical().alias(col) for col in cat_cols]
    return decomposed_dates_df.with_columns(exprs)


def processed_dataframe(encoded_df: pl.DataFrame) -> pl.DataFrame:
    return encoded_df
'''


class ExportService:
    """Package champion models for portable inference."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.registry = RegistryService(workspace)

    def export(
        self,
        key: str,
        output_dir: str | Path | None = None,
        target_col: str | None = None,
    ) -> Path:
        """Export a champion model as a portable directory.

        Args:
            key: Registry key (e.g. "experiment_name:classification").
            output_dir: Output directory. Defaults to workspace/exports/<key>.
            target_col: Target column name for prediction script.

        Returns:
            Path to the export directory.
        """
        entry = self.registry.get(key)
        if entry is None:
            raise ValueError(f"No champion registered for key: {key}")

        model_name = entry["model"]
        artifact_path = entry["artifact_path"]

        if output_dir is None:
            safe_key = key.replace(":", "_").replace("/", "_")
            output_dir = self.workspace.exports_dir / safe_key
        export_path = Path(output_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        if not Path(artifact_path).exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        shutil.copy2(artifact_path, export_path / "model.artifact")

        self._copy_preprocessing(export_path)

        self._write_metadata(key, export_path, model_name, entry, target_col=target_col)

        self._write_predictor(export_path, model_name)

        return export_path

    def _copy_preprocessing(self, export_path: Path) -> None:
        pipelines_dir = export_path / "pipelines"
        pipelines_dir.mkdir(exist_ok=True)
        (pipelines_dir / "preprocessing.py").write_text(PREPROCESSING_TEMPLATE)

        init_path = pipelines_dir / "__init__.py"
        if not init_path.exists():
            init_path.write_text("")

    def _write_metadata(
        self,
        key: str,
        export_path: Path,
        model_name: str,
        entry: dict[str, Any],
        target_col: str | None = None,
    ) -> None:
        from iter8ml.engine.models.factory import _MODEL_REGISTRY

        model_class_info = _MODEL_REGISTRY.get(
            model_name.lower().replace(" ", "_"),
            ("iter8ml.engine.models.catboost_model", "CatBoostModel"),
        )

        task = key.split(":", 1)[1] if ":" in key else "classification"

        metadata = {
            "model_name": model_name,
            "model_class": list(model_class_info),
            "allowlisted_model_classes": [list(v) for v in _MODEL_REGISTRY.values()],
            "task": task,
            "metric": entry.get("metric_name", ""),
            "score": entry.get("score"),
            "run_id": entry.get("run_id"),
            "target_col": target_col or "",
            "registered_at": entry.get("registered_at"),
        }
        with open(export_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _write_predictor(self, export_path: Path, model_name: str) -> None:
        content = PREDICTOR_TEMPLATE.format(model_name=model_name)
        with open(export_path / "predictor.py", "w") as f:
            f.write(content)
