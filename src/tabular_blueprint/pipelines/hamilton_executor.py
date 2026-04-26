"""Hamilton Driver orchestration for data pipelines."""

import polars as pl

from tabular_blueprint.pipelines import preprocessing


class HamiltonExecutor:
    """Orchestrates Hamilton-based data transformations."""

    def __init__(self) -> None:
        try:
            from hamilton import driver

            self.dr = driver.Builder().with_modules(preprocessing).build()
        except ImportError:
            self.dr = None

    def run(self, df: pl.DataFrame, inputs: dict | None = None) -> pl.DataFrame:
        """Run the preprocessing pipeline."""
        if self.dr is None:
            return df

        input_data = {"df": df}
        if inputs:
            input_data.update(inputs)

        result = self.dr.execute(["processed_dataframe"], inputs=input_data)
        return result["processed_dataframe"]

    def get_mermaid_graph(self) -> str:
        if self.dr is None:
            return "graph TD\n    A[Raw Data] --> B[Processed Data]"
        result = self.dr.display_all_functions()
        if isinstance(result, str):
            return result
        return getattr(result, "source", str(result))
