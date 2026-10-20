"""profile_data() and infer_task_spec().

Both operate on the files the platform guarantees for every task (train.csv,
test.csv, data_dictionary.csv, sample_submission.csv) rather than on the
free-text task.md, which is prose for the LLM agent and has no fixed schema.

infer_task_spec() uses data_dictionary.csv's `role` column (id/feature/target)
as the primary source of truth for which columns are which — this was
verified against both provided tasks and held exactly, including an id
column that sample_submission also reproduces. The one thing it cannot read
off the guaranteed files is which metric the task is scored on (the PDF
states private-task metrics "may differ" from the two public ones), so the
metric defaults by task_type and the default is recorded in TaskSpec.notes
instead of being silently assumed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_core.contracts import ColumnProfile, DataProfile, PredictionRequirements, TaskSpec

_DEFAULT_METRIC = {"regression": ("rmse", False), "classification": ("macro_f1", True)}
_HIGH_CARDINALITY_DEFAULT = 20
_TIME_COLUMN_MAX_UNIQUE = 50
_TIME_COLUMN_MIN_OVERLAP_GAP = 0.5  # flag when < 50% of test values seen in train


def load_data_dictionary(path: str | Path) -> pd.DataFrame:
    dd = pd.read_csv(path)
    required = {"feature", "role", "dtype"}
    missing = required - set(dd.columns)
    if missing:
        raise ValueError(f"data_dictionary.csv is missing expected columns: {missing}")
    return dd


def _roles_from_dictionary(dd: pd.DataFrame) -> tuple[list[str], str]:
    id_cols = dd.loc[dd["role"] == "id", "feature"].tolist()
    target_rows = dd.loc[dd["role"] == "target", "feature"].tolist()
    if not id_cols:
        raise ValueError("data_dictionary.csv has no row with role == 'id'")
    if len(target_rows) != 1:
        raise ValueError(
            f"Expected exactly one row with role == 'target' in data_dictionary.csv, found {len(target_rows)}"
        )
    return id_cols, target_rows[0]


def _infer_task_type(target_series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(target_series) or pd.api.types.is_object_dtype(target_series):
        return "classification"
    if isinstance(target_series.dtype, pd.CategoricalDtype):
        return "classification"
    if pd.api.types.is_numeric_dtype(target_series):
        return "regression"
    return "classification"


def infer_task_spec(
    train_path: str | Path,
    data_dictionary_path: str | Path,
    sample_submission_path: str | Path,
    *,
    metric_name: str | None = None,
    greater_is_better: bool | None = None,
) -> TaskSpec:
    """Build a TaskSpec from the platform-guaranteed files.

    Pass metric_name/greater_is_better explicitly whenever you know them
    (e.g. from task.md prose, or because this is one of the two public
    tasks). Left as None, they default by task_type and the default is
    recorded in ``notes`` so nothing is assumed silently.
    """
    dd = load_data_dictionary(data_dictionary_path)
    id_cols, target_col = _roles_from_dictionary(dd)

    train = pd.read_csv(train_path)
    for col in id_cols + [target_col]:
        if col not in train.columns:
            raise ValueError(f"Column {col!r} declared in data_dictionary.csv is absent from train.csv")

    task_type = _infer_task_type(train[target_col])
    notes: list[str] = []

    class_labels: list[str] | None = None
    if task_type == "classification":
        class_labels = sorted(train[target_col].dropna().astype(str).unique().tolist())

    if metric_name is None or greater_is_better is None:
        default_metric, default_direction = _DEFAULT_METRIC[task_type]
        metric_name = metric_name or default_metric
        greater_is_better = default_direction if greater_is_better is None else greater_is_better
        notes.append(
            f"metric_name/greater_is_better defaulted to ({metric_name}, {greater_is_better}) for "
            f"task_type={task_type!r}; confirm against the actual task description before relying on it."
        )

    sub = pd.read_csv(sample_submission_path, nrows=0)
    output_columns = list(sub.columns)
    if target_col not in output_columns:
        notes.append(
            f"sample_submission.csv columns {output_columns} do not contain target_column {target_col!r}; "
            "prediction_requirements.output_columns uses data_dictionary roles instead."
        )
        output_columns = id_cols + [target_col]

    return TaskSpec(
        task_type=task_type,  # type: ignore[arg-type]
        target_column=target_col,
        id_columns=id_cols,
        metric_name=metric_name,
        greater_is_better=greater_is_better,
        class_labels=class_labels,
        prediction_requirements=PredictionRequirements(output_columns=output_columns),
        notes=notes,
    )


def _infer_column_kind(series: pd.Series) -> str:
    n_unique = series.nunique(dropna=True)
    if n_unique <= 1:
        return "constant"
    if pd.api.types.is_bool_dtype(series):
        return "binary"
    if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        # String/category dtype: always categorical, even with exactly two
        # distinct values (e.g. building_class: "Commercial"/"Residential").
        # Only a numeric-dtype column with two values is a true 0/1 flag.
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "binary" if n_unique == 2 else "numeric"
    return "categorical"


def _category_overlap(train_series: pd.Series, test_series: pd.Series) -> float:
    test_vals = set(test_series.dropna().astype(str).unique())
    if not test_vals:
        return 1.0
    train_vals = set(train_series.dropna().astype(str).unique())
    return len(test_vals & train_vals) / len(test_vals)


def _looks_like_time_or_group_column(train_series: pd.Series, test_series: pd.Series, n_unique_train: int) -> bool:
    if n_unique_train > _TIME_COLUMN_MAX_UNIQUE:
        return False
    if not pd.api.types.is_numeric_dtype(train_series):
        return False
    train_vals = set(train_series.dropna().unique())
    test_vals = set(test_series.dropna().unique())
    if not train_vals or not test_vals:
        return False
    overlap = len(train_vals & test_vals) / len(test_vals)
    return overlap < _TIME_COLUMN_MIN_OVERLAP_GAP


def profile_data(
    train_path: str | Path,
    data_dictionary_path: str | Path,
    test_path: str | Path | None = None,
    *,
    high_cardinality_threshold: int = _HIGH_CARDINALITY_DEFAULT,
) -> DataProfile:
    dd = load_data_dictionary(data_dictionary_path)
    id_cols, target_col = _roles_from_dictionary(dd)
    declared_dtype = dict(zip(dd["feature"], dd["dtype"]))

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if test_path is not None else None

    warnings: list[str] = []
    columns: list[ColumnProfile] = []

    for col in train.columns:
        role = "id" if col in id_cols else ("target" if col == target_col else "feature")
        series = train[col]
        kind = _infer_column_kind(series)
        n_unique = int(series.nunique(dropna=True))
        is_high_card = kind == "categorical" and n_unique > high_cardinality_threshold

        overlap = None
        possible_time_or_group = False
        col_warnings: list[str] = []
        declared = declared_dtype.get(col)
        if declared and declared != "unit":
            inferred_matches = (
                (declared == "numeric" and kind in ("numeric", "binary", "constant"))
                or (declared == "category" and kind == "categorical")
                or (declared == "binary" and kind in ("binary", "constant"))
            )
            if not inferred_matches:
                col_warnings.append(
                    f"data_dictionary declares dtype={declared!r} but observed data looks {kind!r}"
                )

        if test is not None and col in test.columns and role == "feature":
            if kind == "categorical":
                overlap = _category_overlap(series, test[col])
                if overlap < 1.0:
                    col_warnings.append(
                        f"test.csv contains categories not seen in train.csv (overlap={overlap:.2%}); "
                        "preprocessing must handle unseen categories, not error on them."
                    )
            possible_time_or_group = _looks_like_time_or_group_column(series, test[col], n_unique)
            if possible_time_or_group:
                col_warnings.append(
                    "low-cardinality numeric column whose train/test value sets barely overlap; "
                    "looks like a time period or group id — consider a time_split/group_kfold validation "
                    "instead of plain k-fold."
                )

        columns.append(
            ColumnProfile(
                name=col,
                role=role,  # type: ignore[arg-type]
                declared_dtype=declared,
                inferred_kind=kind,  # type: ignore[arg-type]
                missing_rate=float(series.isna().mean()),
                n_unique=n_unique,
                is_high_cardinality=is_high_card,
                train_test_category_overlap=overlap,
                possible_time_or_group_column=possible_time_or_group,
                warnings=col_warnings,
            )
        )
        warnings.extend(f"{col}: {w}" for w in col_warnings)

    id_is_unique = bool(train[id_cols].drop_duplicates().shape[0] == train.shape[0])
    if not id_is_unique:
        warnings.append("id column(s) are not unique in train.csv")

    dup_count = int(train.drop(columns=id_cols).duplicated().sum())
    if dup_count:
        warnings.append(
            f"{dup_count} fully duplicated rows in train.csv (excluding id columns); not dropped "
            "automatically — decide deliberately whether they are legitimate repeats or noise."
        )

    target = train[target_col]
    if _infer_task_type(target) == "regression":
        target_summary: dict[str, Any] = {k: float(v) for k, v in target.describe().to_dict().items()}
        target_summary["n_nonpositive"] = int((target <= 0).sum())
        target_summary["n_missing"] = int(target.isna().sum())
    else:
        counts = target.value_counts(dropna=False)
        target_summary = {str(k): int(v) for k, v in counts.to_dict().items()}
        min_class = int(counts.min())
        if min_class < 10:
            warnings.append(
                f"smallest class has only {min_class} examples; cap stratified_kfold n_splits accordingly "
                "and expect noisy per-fold metrics for it."
            )

    return DataProfile(
        n_rows_train=int(train.shape[0]),
        n_rows_test=int(test.shape[0]) if test is not None else None,
        columns=columns,
        duplicated_rows_train=dup_count,
        id_is_unique_train=id_is_unique,
        target_summary=target_summary,
        warnings=warnings,
    )
