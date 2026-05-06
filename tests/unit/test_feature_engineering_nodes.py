from types import SimpleNamespace

import numpy as np

from tabular_blueprint.pipelines.nodes.feature_engineering import _run_embedding


def test_run_embedding_uses_workspace_dir(monkeypatch):
    captured: dict[str, str] = {}

    class DummyEngine:
        def __init__(self, **kwargs):
            captured["workspace_dir"] = kwargs["workspace_dir"]

        def fit_transform(self, **kwargs):
            return kwargs["X"], kwargs["feature_names"]

    monkeypatch.setattr(
        "tabular_blueprint.data.embedding_engine.EmbeddingEngine",
        DummyEngine,
    )

    prep = SimpleNamespace(
        X=np.array([[1.0], [2.0]]),
        y=np.array([0, 1]),
        feature_names=["f1"],
        _df=None,
    )
    _run_embedding(
        data_prep_result=prep,
        task="classification",
        random_seed=42,
        run_id="exp_1",
        workspace_dir="/tmp/custom-workspace",
    )

    assert captured["workspace_dir"] == "/tmp/custom-workspace"
