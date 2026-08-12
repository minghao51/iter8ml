"""Bundled demo datasets shipped with iter8ml.

Small, recognizable tabular datasets so ``iter8 init --demo`` gives a
zero-friction first run with no download step. See ``README.md`` in this
directory for sources and attribution.
"""

from pathlib import Path

_DATASETS_DIR = Path(__file__).parent

#: Mapping of demo dataset name -> bundled parquet filename.
_BUNDLED: dict[str, str] = {
    "telco_churn": "telco_churn.parquet",
}

__all__ = ["BUNDLED_DATASETS", "bundled_dataset_path"]

#: Read-only tuple of available bundled dataset names.
BUNDLED_DATASETS: tuple[str, ...] = tuple(sorted(_BUNDLED))


def bundled_dataset_path(name: str) -> Path:
    """Return the absolute path to a bundled demo dataset parquet.

    Args:
        name: Dataset key (e.g. ``"telco_churn"``).

    Returns:
        Absolute :class:`~pathlib.Path` to the bundled ``.parquet`` file.

    Raises:
        KeyError: If ``name`` is not a bundled dataset.
    """
    if name not in _BUNDLED:
        available = ", ".join(sorted(_BUNDLED))
        raise KeyError(f"Unknown bundled dataset {name!r}. Available: {available}")
    return _DATASETS_DIR / _BUNDLED[name]
