"""Integration test: full Hamilton DAG path via PipelineExecutor directly."""

import polars as pl
import pytest
from sklearn.datasets import make_classification

from iter8ml.config import ExperimentConfig, PipelineSpec, PipelineStep, StepName
from iter8ml.engine.pipelines.executor import PipelineExecutor, PipelineMode
from iter8ml.engine.tracker import JSONLTracker
from iter8ml.workspace import Workspace

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not pytest.importorskip("hamilton"),
        reason="Hamilton not installed",
    ),
]


@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    return df.with_columns(target=pl.Series(y))


@pytest.fixture
def config(tmp_path):
    return ExperimentConfig(
        name="dag_integration_test",
        task="classification",
        target_col="target",
        data_path="",
        cv_folds=3,
        metrics=["roc_auc", "f1_macro"],
        models=["catboost"],
        pipeline=PipelineSpec(
            steps=[
                PipelineStep(name=StepName.DATA_PREP),
                PipelineStep(name=StepName.QUALITY_AUDIT, enabled=False),
                PipelineStep(name=StepName.LEAKAGE_AUDIT, enabled=False),
                PipelineStep(name=StepName.TARGET_TRANSFORM),
                PipelineStep(name=StepName.FEATURE_ENGINEERING),
                PipelineStep(name=StepName.MODEL_TRAINING),
                PipelineStep(name=StepName.CALIBRATION),
                PipelineStep(name=StepName.EVALUATION),
            ]
        ),
    )


class TestDAGExecution:
    def test_run_training_returns_training_state(self, config, classification_data, tmp_path):
        executor = PipelineExecutor(mode=PipelineMode.TRAINING)
        state = executor.run_training(
            config=config,
            df=classification_data,
            run_id="test_dag_001",
            vram_gb=0.0,
            workspace=Workspace(root=tmp_path),
        )

        assert state is not None
        assert hasattr(state, "results")
        assert hasattr(state, "best_model")
        assert hasattr(state, "best_score")

    def test_run_training_produces_model_results(self, config, classification_data, tmp_path):
        executor = PipelineExecutor(mode=PipelineMode.TRAINING)
        state = executor.run_training(
            config=config,
            df=classification_data,
            run_id="test_dag_002",
            vram_gb=0.0,
            workspace=Workspace(root=tmp_path),
        )

        assert "catboost" in state.results
        catboost_result = state.results["catboost"]
        assert "cv_scores" in catboost_result
        assert "roc_auc" in catboost_result["cv_scores"]
        assert catboost_result["cv_scores"]["roc_auc"] > 0.5
        assert "duration_seconds" in catboost_result
        assert catboost_result["duration_seconds"] > 0

    def test_run_training_with_tracking_hook(self, config, classification_data, tmp_path):
        tracker = JSONLTracker(str(tmp_path / "experiments.jsonl"))
        executor = PipelineExecutor(mode=PipelineMode.TRAINING, tracker=tracker)
        state = executor.run_training(
            config=config,
            df=classification_data,
            run_id="test_dag_003",
            vram_gb=0.0,
            workspace=Workspace(root=tmp_path),
        )

        assert state is not None
        assert tmp_path.exists()

    def test_run_training_best_model_is_selected(self, config, classification_data, tmp_path):
        multi_config = config.model_copy(update={"models": ["catboost", "lightgbm"]})
        executor = PipelineExecutor(mode=PipelineMode.TRAINING)
        state = executor.run_training(
            config=multi_config,
            df=classification_data,
            run_id="test_dag_004",
            vram_gb=0.0,
            workspace=Workspace(root=tmp_path),
        )

        assert state.best_model is not None
        assert state.best_score is not None
        assert state.best_score > 0.5
        assert len(state.leaderboard) >= 1

    def test_leaderboard_sorted_by_score(self, config, classification_data, tmp_path):
        multi_config = config.model_copy(update={"models": ["catboost", "lightgbm", "xgboost"]})
        executor = PipelineExecutor(mode=PipelineMode.TRAINING)
        state = executor.run_training(
            config=multi_config,
            df=classification_data,
            run_id="test_dag_005",
            vram_gb=0.0,
            workspace=Workspace(root=tmp_path),
        )

        scores = [entry["score"] for entry in state.leaderboard]
        assert scores == sorted(scores, reverse=True)
