"""create_validation(): build and persist a reusable, honest validation split.

A ValidationPlan stores only row *positions* (0-based, matching the row order
of the train.csv it was built from) in an .npz file, plus a fingerprint of
that train.csv. training.run_experiment() re-checks the fingerprint before
using a plan, so an experiment can never be silently scored against a split
built for different data — and two ExperimentResults are only safe to compare
head-to-head when they share the same validation_id (see
ExperimentResult.is_fully_comparable and experiment_log.select_best).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from ml_core.contracts import TaskSpec, ValidationPlan, ValidationStrategy
from ml_core.errors import ValidationMismatchError, ValidationPlanError
from ml_core.io_utils import config_hash, ensure_dir, file_fingerprint

_MIN_CLASS_COUNT_FOR_STRATIFICATION = 2


def _default_strategy(task_spec: TaskSpec) -> ValidationStrategy:
    return "stratified_kfold" if task_spec.task_type == "classification" else "kfold"


def _build_kfold_indices(n_rows: int, n_splits: int, seed: int) -> list[np.ndarray]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [val_idx for _, val_idx in kf.split(np.arange(n_rows))]


def _build_stratified_indices(y: pd.Series, n_splits: int, seed: int) -> tuple[list[np.ndarray], list[str]]:
    counts = y.value_counts()
    min_count = int(counts.min())
    warnings: list[str] = []
    if min_count < _MIN_CLASS_COUNT_FOR_STRATIFICATION:
        raise ValidationPlanError(
            f"Class {counts.idxmin()!r} has only {min_count} example(s); StratifiedKFold needs at least "
            f"{_MIN_CLASS_COUNT_FOR_STRATIFICATION} per class per fold. Use strategy='kfold' instead, or "
            "collect/merge rare classes before validating."
        )
    effective_splits = min(n_splits, min_count)
    if effective_splits < n_splits:
        warnings.append(
            f"n_splits reduced from {n_splits} to {effective_splits}: the rarest class "
            f"({counts.idxmin()!r}, n={min_count}) cannot support more folds than it has members."
        )
    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
    folds = [val_idx for _, val_idx in skf.split(np.zeros(len(y)), y)]
    return folds, warnings


def _build_group_indices(groups: pd.Series, n_splits: int) -> list[np.ndarray]:
    n_groups = groups.nunique()
    if n_groups < n_splits:
        raise ValidationPlanError(
            f"group_column has only {n_groups} distinct group(s), cannot build {n_splits} GroupKFold splits."
        )
    gkf = GroupKFold(n_splits=n_splits)
    return [val_idx for _, val_idx in gkf.split(np.arange(len(groups)), groups=groups)]


def _build_time_split_indices(time_values: pd.Series, n_splits: int) -> list[np.ndarray]:
    ordered_unique = np.sort(time_values.dropna().unique())
    if len(ordered_unique) < n_splits + 1:
        raise ValidationPlanError(
            f"time_column has only {len(ordered_unique)} distinct value(s); cannot reserve {n_splits} of "
            "them as held-out future periods while leaving at least one earlier period to train on."
        )
    holdout_values = ordered_unique[-n_splits:]
    positions = np.arange(len(time_values))
    return [positions[(time_values == v).to_numpy()] for v in holdout_values]


def create_validation(
    task_spec: TaskSpec,
    train_path: str | Path,
    out_dir: str | Path,
    *,
    strategy: ValidationStrategy | None = None,
    n_splits: int = 5,
    seed: int = 42,
    time_column: str | None = None,
    group_column: str | None = None,
) -> ValidationPlan:
    strategy = strategy or _default_strategy(task_spec)
    if strategy == "time_split" and not time_column:
        raise ValidationPlanError("strategy='time_split' requires time_column")
    if strategy == "group_kfold" and not group_column:
        raise ValidationPlanError("strategy='group_kfold' requires group_column")

    usecols = [task_spec.target_column]
    if time_column:
        usecols.append(time_column)
    if group_column:
        usecols.append(group_column)
    df = pd.read_csv(train_path, usecols=sorted(set(usecols)))
    n_rows = len(df)

    warnings: list[str] = []
    if strategy == "stratified_kfold":
        folds, w = _build_stratified_indices(df[task_spec.target_column], n_splits, seed)
        warnings.extend(w)
    elif strategy == "kfold":
        folds = _build_kfold_indices(n_rows, n_splits, seed)
    elif strategy == "group_kfold":
        folds = _build_group_indices(df[group_column], n_splits)
    elif strategy == "time_split":
        folds = _build_time_split_indices(df[time_column], n_splits)
    else:
        raise ValidationPlanError(f"Unknown validation strategy {strategy!r}")

    train_fingerprint = file_fingerprint(train_path)
    payload_for_id = {
        "strategy": strategy,
        "n_splits": len(folds),
        "seed": seed,
        "time_column": time_column,
        "group_column": group_column,
        "train_fingerprint": train_fingerprint,
    }
    validation_id = config_hash(payload_for_id)

    out_dir = ensure_dir(out_dir)
    fold_indices_path = out_dir / f"validation_{validation_id}.npz"
    np.savez(fold_indices_path, n_rows=n_rows, **{f"fold_{i}": idx for i, idx in enumerate(folds)})

    return ValidationPlan(
        validation_id=validation_id,
        strategy=strategy,
        n_splits=len(folds),
        seed=seed,
        train_fingerprint=train_fingerprint,
        time_column=time_column,
        group_column=group_column,
        fold_indices_path=str(fold_indices_path),
        warnings=warnings,
    )


def load_fold_indices(plan: ValidationPlan, train_path: str | Path) -> list[np.ndarray]:
    """Load persisted fold validation-indices, re-checking the train fingerprint first."""
    current_fingerprint = file_fingerprint(train_path)
    if current_fingerprint != plan.train_fingerprint:
        raise ValidationMismatchError(
            f"train.csv fingerprint {current_fingerprint} does not match the fingerprint "
            f"{plan.train_fingerprint} this ValidationPlan was built from. Rebuild the plan with "
            "create_validation() before reusing it."
        )
    with np.load(plan.fold_indices_path) as npz:
        n_rows = int(npz["n_rows"])
        folds = [npz[f"fold_{i}"] for i in range(plan.n_splits)]
    train_n_rows = sum(1 for _ in open(train_path, "r", encoding="utf-8")) - 1
    if train_n_rows != n_rows:
        raise ValidationMismatchError(
            f"train.csv has {train_n_rows} rows but this ValidationPlan was built for {n_rows} rows."
        )
    return folds
