"""Shared pytest fixtures for integration tests."""

import polars as pl
import pytest
from sklearn.datasets import make_classification, make_regression


@pytest.fixture(scope="session")
def classification_data():
    """Small classification dataset for quick integration tests."""
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    return df.with_columns(target=pl.Series(y))


@pytest.fixture(scope="session")
def regression_data():
    """Small regression dataset for quick integration tests."""
    X, y = make_regression(n_samples=500, n_features=10, n_informative=5, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    return df.with_columns(target=pl.Series(y))


@pytest.fixture(scope="session")
def large_classification_data():
    """Larger classification dataset (>10k rows) to test model routing."""
    X, y = make_classification(n_samples=15000, n_features=20, n_informative=10, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    return df.with_columns(target=pl.Series(y))


@pytest.fixture
def tmp_workspace(tmp_path):
    """Temporary workspace directory for experiment outputs."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace
