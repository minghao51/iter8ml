"""Pre-run data/config sanity checks: catch wasted runs before training starts.

``run_preflight`` inspects the actual DataFrame against the resolved config and
returns severity-ranked issues. It is deliberately side-effect free: no
workspace writes, no model fits — the point is to fail fast on misconfigurations
that would otherwise burn full prep/feature-engineering compute (or worse,
produce a plausible-looking but wrong report).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import polars as pl

from iter8ml.config import ExperimentConfig
from iter8ml.constants import CVStrategy, TaskType

# Column-name hints that suggest identifier columns (unique per row) which are
# almost always leakage when used as features.
_ID_NAME_HINTS: tuple[str, ...] = ("id", "uuid", "guid", "key", "index")


class IssueLevel(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass
class PreflightIssue:
    level: IssueLevel
    check: str
    message: str

    def format(self) -> str:
        return f"[{self.level.value.upper():7s}] ({self.check}) {self.message}"


def run_preflight(config: ExperimentConfig, df: pl.DataFrame) -> list[PreflightIssue]:
    """Validate ``config`` against ``df``; return issues (errors fail the run)."""
    issues: list[PreflightIssue] = []
    issues.extend(_check_columns(config, df))
    issues.extend(_check_target(config, df))
    issues.extend(_check_cv_feasibility(config, df))
    issues.extend(_check_feature_quality(config, df))
    return issues


def has_errors(issues: list[PreflightIssue]) -> bool:
    return any(i.level == IssueLevel.ERROR for i in issues)


def _check_columns(config: ExperimentConfig, df: pl.DataFrame) -> list[PreflightIssue]:
    issues: list[PreflightIssue] = []
    unknown = [c for c in config.ignore_cols if c not in df.columns]
    if unknown:
        issues.append(
            PreflightIssue(
                IssueLevel.ERROR,
                "ignore_cols",
                f"columns not found in data: {unknown}. Available: {df.columns}",
            )
        )
    if config.target_col in config.ignore_cols:
        issues.append(
            PreflightIssue(
                IssueLevel.ERROR,
                "ignore_cols",
                f"target_col '{config.target_col}' cannot also be listed in ignore_cols",
            )
        )
    return issues


def _check_target(config: ExperimentConfig, df: pl.DataFrame) -> list[PreflightIssue]:
    issues: list[PreflightIssue] = []
    target = config.target_col

    if target not in df.columns:
        issues.append(
            PreflightIssue(
                IssueLevel.ERROR,
                "target",
                f"target column '{target}' not found in data. Available: {df.columns}",
            )
        )
        return issues

    series = df[target]
    n_nulls = series.null_count()
    if n_nulls:
        issues.append(
            PreflightIssue(
                IssueLevel.ERROR,
                "target",
                f"target column '{target}' contains {n_nulls} null value(s) "
                f"({n_nulls / len(df):.1%}); drop or impute them before training",
            )
        )

    n_unique = series.n_unique()
    if n_unique <= 1:
        issues.append(
            PreflightIssue(
                IssueLevel.ERROR,
                "target",
                f"target column '{target}' is constant (n_unique={n_unique}); "
                "there is nothing to predict",
            )
        )
        return issues

    if config.positive_class is not None:
        values = series.unique().to_list()
        if config.positive_class not in values:
            issues.append(
                PreflightIssue(
                    IssueLevel.ERROR,
                    "target",
                    f"positive_class {config.positive_class!r} not found in target "
                    f"'{target}'. Observed values: {sorted(str(v) for v in values)}",
                )
            )
        elif len(values) > 2:
            issues.append(
                PreflightIssue(
                    IssueLevel.ERROR,
                    "target",
                    f"positive_class requires a binary target; '{target}' has "
                    f"{len(values)} distinct values",
                )
            )
        else:
            issues.append(
                PreflightIssue(
                    IssueLevel.WARNING,
                    "target",
                    f"positive_class={config.positive_class!r}: the positive class "
                    "will be encoded to 1 so probability metrics are oriented explicitly",
                )
            )

    if config.task == TaskType.CLASSIFICATION:
        if n_unique > 20 and series.dtype.is_float():
            issues.append(
                PreflightIssue(
                    IssueLevel.WARNING,
                    "target",
                    f"task is classification but target '{target}' has {n_unique} "
                    "distinct float values; this looks like a regression target — "
                    "use task: regression if so",
                )
            )
        min_class_share = _min_class_share(series)
        if min_class_share is not None and min_class_share < 0.01:
            issues.append(
                PreflightIssue(
                    IssueLevel.WARNING,
                    "imbalance",
                    f"rarest class holds {min_class_share:.2%} of rows; with no "
                    "class-weighting support, consider model_overrides (e.g. "
                    "scale_pos_weight) or resampling before trusting roc_auc",
                )
            )
    else:
        if not series.dtype.is_numeric():
            issues.append(
                PreflightIssue(
                    IssueLevel.ERROR,
                    "target",
                    f"task is regression but target '{target}' has non-numeric dtype "
                    f"'{series.dtype}'; encode it or fix the task type",
                )
            )
        if n_unique < 20:
            issues.append(
                PreflightIssue(
                    IssueLevel.WARNING,
                    "target",
                    f"regression target '{target}' has only {n_unique} distinct values; "
                    "this may be a classification target misdeclared as regression",
                )
            )
    return issues


def _check_cv_feasibility(config: ExperimentConfig, df: pl.DataFrame) -> list[PreflightIssue]:
    issues: list[PreflightIssue] = []
    if config.cv_folds < 2:
        return issues
    if len(df) < config.cv_folds * 2:
        issues.append(
            PreflightIssue(
                IssueLevel.ERROR,
                "cv",
                f"{config.cv_folds} folds need at least ~{config.cv_folds * 2} rows "
                f"for both train and validation per fold; dataset has {len(df)}",
            )
        )
    if config.task == TaskType.CLASSIFICATION and config.target_col in df.columns:
        min_class_count = _min_class_count(df[config.target_col])
        if (
            config.cv_strategy == CVStrategy.STRATIFIED
            and min_class_count is not None
            and min_class_count < config.cv_folds
        ):
            issues.append(
                PreflightIssue(
                    IssueLevel.ERROR,
                    "cv",
                    f"stratified {config.cv_folds}-fold CV needs >= {config.cv_folds} "
                    f"rows in every class; rarest class has {min_class_count}",
                )
            )
    if config.cv_strategy == CVStrategy.TIMESERIES:
        date_cols = [c for c, dtype in df.schema.items() if dtype in (pl.Datetime, pl.Date)]
        if not date_cols:
            issues.append(
                PreflightIssue(
                    IssueLevel.WARNING,
                    "cv",
                    "cv_strategy='timeseries' but data has no date/datetime column; "
                    "ensure rows are already in chronological order or folds leak",
                )
            )
    return issues


def _check_feature_quality(config: ExperimentConfig, df: pl.DataFrame) -> list[PreflightIssue]:
    issues: list[PreflightIssue] = []
    skip = {config.target_col, *config.ignore_cols}
    for col in df.columns:
        if col in skip:
            continue
        if df[col].null_count() == len(df):
            issues.append(
                PreflightIssue(IssueLevel.WARNING, "features", f"column '{col}' is entirely null")
            )
        if df[col].n_unique() == len(df) and len(df) > 20:
            hinted = any(h in col.lower() for h in _ID_NAME_HINTS)
            if hinted:
                issues.append(
                    PreflightIssue(
                        IssueLevel.WARNING,
                        "leakage",
                        f"column '{col}' is unique per row and name suggests an ID; "
                        "add it to ignore_cols unless it is genuinely predictive",
                    )
                )
    return issues


def _min_class_share(series: pl.Series) -> float | None:
    counts = series.drop_nulls().value_counts()
    if counts.height == 0 or len(series) == 0:
        return None
    min_count = min(counts["count"].to_list())
    return float(min_count) / float(len(series))


def _min_class_count(series: pl.Series) -> int | None:
    counts = series.drop_nulls().value_counts()
    if counts.height == 0:
        return None
    return int(min(counts["count"].to_list()))
