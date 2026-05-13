"""Tests for Workspace abstraction."""

from pathlib import Path

from iter8ml.workspace import Workspace, _default_root


def test_default_root_no_env():
    assert _default_root() == Path("workspace")


def test_default_root_with_env(monkeypatch):
    monkeypatch.setenv("ITER8ML_WORKSPACE", "/my/custom/path")
    assert _default_root() == Path("/my/custom/path")


def test_init_with_root_path(tmp_path):
    ws = Workspace(root=tmp_path)
    assert ws.root == tmp_path


def test_init_converts_string_to_path(tmp_path):
    ws = Workspace(root=str(tmp_path))
    assert isinstance(ws.root, Path)
    assert ws.root == tmp_path


def test_default_root_uses_default():
    ws = Workspace()
    assert ws.root == Path("workspace")


def test_experiments_path(tmp_path):
    ws = Workspace(root=tmp_path)
    assert ws.experiments_path == tmp_path / "experiments.jsonl"


def test_registry_path(tmp_path):
    ws = Workspace(root=tmp_path)
    assert ws.registry_path == tmp_path / "registry.json"


def test_artifacts_dir(tmp_path):
    ws = Workspace(root=tmp_path)
    assert ws.artifacts_dir == tmp_path / "artifacts"


def test_exports_dir(tmp_path):
    ws = Workspace(root=tmp_path)
    assert ws.exports_dir == tmp_path / "exports"


def test_state_path(tmp_path):
    ws = Workspace(root=tmp_path)
    assert ws.state_path == tmp_path / "current_state.md"


def test_leaderboard_path(tmp_path):
    ws = Workspace(root=tmp_path)
    assert ws.leaderboard_path == tmp_path / "leaderboard.md"


def test_init_creates_directories(tmp_path):
    ws = Workspace(root=tmp_path)
    ws.init()
    assert ws.artifacts_dir.exists()
    assert ws.exports_dir.exists()


def test_init_creates_experiments_file(tmp_path):
    ws = Workspace(root=tmp_path)
    ws.init()
    assert ws.experiments_path.exists()


def test_init_creates_registry_file(tmp_path):
    ws = Workspace(root=tmp_path)
    ws.init()
    assert ws.registry_path.exists()
    assert ws.registry_path.read_text() == "{}"


def test_init_preserves_existing_registry(tmp_path):
    ws = Workspace(root=tmp_path)
    ws.registry_path.parent.mkdir(parents=True, exist_ok=True)
    ws.registry_path.write_text('{"catboost": "champion"}')
    ws.init()
    assert ws.registry_path.read_text() == '{"catboost": "champion"}'


def test_init_returns_self(tmp_path):
    ws = Workspace(root=tmp_path)
    result = ws.init()
    assert result is ws


def test_init_idempotent(tmp_path):
    ws = Workspace(root=tmp_path)
    ws.init()
    ws.init()
    assert ws.artifacts_dir.exists()
    assert ws.exports_dir.exists()
