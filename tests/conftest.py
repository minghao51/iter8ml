import polars as pl
import pytest
from sklearn.datasets import make_classification, make_regression


@pytest.fixture(scope="session")
def classification_data():
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    return df.with_columns(target=pl.Series(y))


@pytest.fixture(scope="session")
def regression_data():
    X, y = make_regression(n_samples=500, n_features=10, n_informative=5, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    return df.with_columns(target=pl.Series(y))


@pytest.fixture
def tmp_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


def pytest_collection_modifyitems(items):
    for item in items:
        if "unit/" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "integration/" in item.nodeid:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
        elif "e2e/" in item.nodeid:
            item.add_marker(pytest.mark.e2e)
            item.add_marker(pytest.mark.slow)
