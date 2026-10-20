"""Metric registry.

The metric for a task is fixed in TaskSpec (metric_name + greater_is_better)
and is never read from ExperimentConfig — an experiment is not allowed to
pick its own evaluator. training.run_experiment always resolves the metric
through get_metric(task_spec.metric_name) and cross-checks
greater_is_better against the registry, so a mismatched TaskSpec fails loudly
instead of silently flipping the comparison direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from ml_core.errors import UnknownMetricError

if TYPE_CHECKING:
    from ml_core.contracts import TaskSpec


@dataclass(frozen=True)
class Metric:
    name: str
    greater_is_better: bool
    fn: Callable[[np.ndarray, np.ndarray], float]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


_REGISTRY: dict[str, Metric] = {
    "rmse": Metric("rmse", greater_is_better=False, fn=_rmse),
    "mae": Metric("mae", greater_is_better=False, fn=_mae),
    "r2": Metric("r2", greater_is_better=True, fn=_r2),
    "macro_f1": Metric("macro_f1", greater_is_better=True, fn=_macro_f1),
    "weighted_f1": Metric("weighted_f1", greater_is_better=True, fn=_weighted_f1),
    "accuracy": Metric("accuracy", greater_is_better=True, fn=_accuracy),
}

# Metrics computed for context even when they are not the competition metric.
REGRESSION_AUX_METRICS = ("mae", "r2")
CLASSIFICATION_AUX_METRICS = ("accuracy", "weighted_f1")


def get_metric(name: str) -> Metric:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY))
        raise UnknownMetricError(f"Unknown metric {name!r}. Known metrics: {known}") from exc


def available_metrics() -> list[str]:
    return sorted(_REGISTRY)


def build_task_metric_fn(task_spec: "TaskSpec") -> Callable[[np.ndarray, np.ndarray], float]:
    """Metric function bound to the task's full, fixed label set.

    For classification this matters more than it looks: sklearn's
    f1_score(average="macro") without an explicit `labels=` argument only
    averages over classes that actually appear in the y_true/y_pred given to
    it. On a small CV fold a rare class (this competition has one with only
    65 training rows total) can be absent by chance, which would silently
    average over fewer classes and read as a better score than it should.
    Binding `labels` to every class in task_spec.class_labels makes every
    fold's macro F1 computed over the same fixed denominator.
    """
    metric = get_metric(task_spec.metric_name)
    if metric.greater_is_better != task_spec.greater_is_better:
        raise UnknownMetricError(
            f"TaskSpec.greater_is_better={task_spec.greater_is_better} disagrees with the registered "
            f"direction for metric {task_spec.metric_name!r} ({metric.greater_is_better}). "
            "This looks like a contract bug, not a modeling choice — fix TaskSpec."
        )
    if task_spec.task_type == "classification" and task_spec.metric_name in ("macro_f1", "weighted_f1"):
        average = "macro" if task_spec.metric_name == "macro_f1" else "weighted"
        n_classes = len(task_spec.class_labels or [])
        return partial(f1_score, average=average, labels=list(range(n_classes)), zero_division=0)
    return metric.fn
