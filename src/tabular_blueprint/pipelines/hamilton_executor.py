import warnings

import polars as pl

from tabular_blueprint.pipelines.executor import PipelineExecutor


class HamiltonExecutor:
    def __init__(self) -> None:
        self._executor = PipelineExecutor()

    def run(self, df: pl.DataFrame, inputs: dict | None = None) -> pl.DataFrame:
        warnings.warn(
            "HamiltonExecutor is deprecated. Use PipelineExecutor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self._executor.available:
            return df
        return self._executor.run_preprocessing(df)

    def get_mermaid_graph(self) -> str:
        return self._executor.get_mermaid_graph()
