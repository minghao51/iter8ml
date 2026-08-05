#!/usr/bin/env python3
"""CLI entry point for the OpenML benchmark suite.

Usage::

    uv run python benchmarks/run_openml_benchmark.py [--quick]
    uv run python benchmarks/run_openml_benchmark.py \
        --sweep-config sweeps/catboost_task_type.yaml --quick
    uv run python benchmarks/run_openml_benchmark.py --save-baseline --quick
    uv run python benchmarks/run_openml_benchmark.py \
        --check-regression benchmarks/results/baseline_summary.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.openml_benchmark import main
from iter8ml.config import HardwareProfile

# Configure OpenMP threads via the framework's own (Linux-capped) default BEFORE
# any C-extension ML lib (lightgbm/xgboost) loads libgomp. On hybrid-core (P+E)
# CPUs under WSL2/Linux, libgomp deadlocks across all cores; the framework caps at
# 8 on Linux (override via OMP_NUM_THREADS).
HardwareProfile.configure_omp_threads()

if __name__ == "__main__":
    main()
