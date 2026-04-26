import polars as pl
import pytest
from sklearn.datasets import make_classification


@pytest.fixture(scope="session")
def large_classification_data():
    X, y = make_classification(n_samples=15000, n_features=20, n_informative=10, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    return df.with_columns(target=pl.Series(y))
