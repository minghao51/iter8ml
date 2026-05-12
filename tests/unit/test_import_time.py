"""Smoke test: verify core import path completes quickly without optional deps."""

import subprocess
import sys

import pytest

_HAS_TORCH = pytest.importorskip("importlib").util.find_spec("torch") is not None


class TestImportTime:
    """Ensure importing the package with only core deps is fast and safe."""

    def test_import_completes_in_one_second(self) -> None:
        """Subprocess isolates us from test-runner's already-loaded modules."""
        cmd = [
            sys.executable,
            "-c",
            (
                "import time, sys; "
                "t0 = time.perf_counter(); "
                "import iter8ml; "
                "elapsed = time.perf_counter() - t0; "
                "print(f'import_time={elapsed:.3f}s'); "
                "sys.exit(0 if elapsed < 1.0 else 1)"
            ),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (
            f"Import took >= 1.0s or failed. stdout={result.stdout!r} stderr={result.stderr!r}"
        )

    @pytest.mark.skipif(_HAS_TORCH, reason="torch installed in this environment")
    def test_deep_models_fail_gracefully_without_torch(self) -> None:
        """FT-Transformer should raise ImportError when torch is unavailable."""
        # The module should be importable even without torch installed.
        # The error only happens when you try to instantiate the model.
        from iter8ml.engine.models.ft_transformer import FTTransformerModel

        with pytest.raises(ImportError):
            FTTransformerModel()

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed in this environment")
    def test_deep_models_construct_when_torch_present(self) -> None:
        """FT-Transformer should construct successfully when torch is available."""
        from iter8ml.engine.models.ft_transformer import FTTransformerModel

        model = FTTransformerModel()
        assert model.model_name == "FT-Transformer"

    def test_mcp_tools_import_without_fastmcp(self) -> None:
        """mcp.tools should be importable even if mcp package is missing."""
        # The tool functions are plain Python functions — only accessing ``mcp``
        # triggers the lazy FastMCP initializer.
        from iter8ml.services.mcp import (
            get_column_stats,
            get_event_log,
            get_experiment_state,
        )

        assert callable(get_experiment_state)
        assert callable(get_column_stats)
        assert callable(get_event_log)
