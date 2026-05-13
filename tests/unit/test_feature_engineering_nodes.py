from types import SimpleNamespace

import numpy as np

from iter8ml.engine.pipelines.nodes.features import _run_embedding
from iter8ml.workspace import Workspace


def test_run_embedding_uses_workspace(monkeypatch):
    captured: dict[str, object] = {}

    class DummyEngine:
        def __init__(self, **kwargs):
            captured["workspace"] = kwargs["workspace"]

        def fit_transform(self, **kwargs):
            return kwargs["X"], kwargs["feature_names"]

    monkeypatch.setattr(
        "iter8ml.data.embedding.EmbeddingEngine",
        DummyEngine,
    )

    prep = SimpleNamespace(
        X=np.array([[1.0], [2.0]]),
        y=np.array([0, 1]),
        feature_names=["f1"],
        _df=None,
    )
    ws = Workspace(root="/tmp/custom-workspace")
    _run_embedding(
        data_prep_result=prep,
        task="classification",
        random_seed=42,
        run_id="exp_1",
        workspace=ws,
    )

    assert captured["workspace"] is ws
