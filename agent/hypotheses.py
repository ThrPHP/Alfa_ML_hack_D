"""Turn hypotheses into ml_core ExperimentConfigs, plus the no-LLM playbook.

A Hypothesis (from the LLM, or from the deterministic playbook below) is a
*change* relative to the current best config (base="best") or a config from
scratch (base="scratch"). materialize() applies it, sanitizes
hyperparameters (threads/seeds/verbosity are ours to set, iteration counts
and tree sizes are clamped so one hypothesis cannot eat the whole budget),
and builds a strict ExperimentConfig whose experiment_id is a stable hash —
the same idea proposed twice maps to the same id and is skipped.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ml_core import model_registry
from ml_core.contracts import DataProfile, ExperimentConfig, PreprocessingConfig, TaskSpec
from ml_core.io_utils import config_hash

from agent.schemas import Hypothesis


class HypothesisError(Exception):
    pass


_RESERVED_KEYS = {
    "n_jobs", "thread_count", "nthread", "num_threads", "verbose", "verbosity", "silent", "logging_level",
    "random_state", "random_seed", "seed", "cat_features", "task_type", "device", "devices", "gpu_id",
    "callbacks", "early_stopping_rounds", "od_type", "od_wait", "use_best_model", "eval_set",
    "train_dir", "allow_writing_files", "categorical_features", "enable_categorical", "class_labels",
}
_ITERATION_KEYS = ("n_estimators", "iterations", "num_boost_round", "num_iterations", "max_iter")
_MAX_ITERATIONS = 4000
_CLAMPS: dict[str, tuple[float, float]] = {
    "learning_rate": (0.005, 0.5),
    "num_leaves": (4, 1024),
    "depth": (1, 10),
    "max_depth": (-1, 16),
    "min_child_samples": (1, 2000),
}


def sanitize_hyperparameters(model_name: str, hyperparameters: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    out: dict[str, Any] = {}
    for key, value in hyperparameters.items():
        if key in _RESERVED_KEYS:
            notes.append(f"ignored reserved hyperparameter {key!r}")
            continue
        if key in _ITERATION_KEYS and isinstance(value, (int, float)) and value > _MAX_ITERATIONS:
            notes.append(f"{key} clamped {value} -> {_MAX_ITERATIONS}")
            value = _MAX_ITERATIONS
        if key in _CLAMPS and isinstance(value, (int, float)) and not isinstance(value, bool):
            lo, hi = _CLAMPS[key]
            clamped = min(max(value, lo), hi)
            if clamped != value:
                notes.append(f"{key} clamped {value} -> {clamped}")
                value = type(value)(clamped)
        out[key] = value
    if model_name == "catboost":
        out["allow_writing_files"] = False  # otherwise catboost_info/ lands in the cwd
    return out, notes


def stable_experiment_id(config: ExperimentConfig) -> str:
    return config_hash(config.model_dump(mode="json", exclude={"experiment_id", "time_budget_s", "n_jobs"}))


def materialize(
    hypothesis: Hypothesis,
    best: ExperimentConfig | None,
    validation_id: str,
    *,
    n_jobs: int | None = -1,
    seed: int = 42,
    extra_feature_steps: list[str] | None = None,
) -> tuple[ExperimentConfig, list[str]]:
    """Build the ExperimentConfig for a hypothesis. Raises HypothesisError with a readable reason."""
    start = best if (hypothesis.base == "best" and best is not None) else None
    model_name = hypothesis.model_name or (start.model_name if start else None)
    if not model_name:
        raise HypothesisError("hypothesis has no model_name and there is no best config to start from")
    if model_name not in model_registry.ALL_MODELS:
        raise HypothesisError(f"unknown model_name {model_name!r}; choose from {sorted(model_registry.ALL_MODELS)}")

    same_model = start is not None and start.model_name == model_name
    hyper = {**(start.hyperparameters if same_model else {}), **hypothesis.hyperparameters}
    hyper, notes = sanitize_hyperparameters(model_name, hyper)

    pre: dict[str, Any] = start.preprocessing.model_dump() if start else {}
    if not same_model:
        pre["mode"] = model_registry.suggested_mode(model_name)
    pre.update(hypothesis.preprocessing)
    if model_registry.requires_linear_mode(model_name) and pre.get("mode") != "linear":
        pre["mode"] = "linear"
        notes.append(f"preprocessing.mode forced to 'linear' for {model_name}")

    steps = list(hypothesis.feature_steps) if hypothesis.feature_steps is not None else list(start.feature_steps if start else [])
    for step in [*hypothesis.add_feature_steps, *(extra_feature_steps or [])]:
        if step not in steps:
            steps.append(step)

    try:
        config = ExperimentConfig(
            model_name=model_name,
            hyperparameters=hyper,
            preprocessing=PreprocessingConfig(**pre),
            feature_steps=steps,
            random_seed=seed,
            validation_id=validation_id,
            n_jobs=n_jobs,
        )
    except ValidationError as exc:
        raise HypothesisError(f"invalid config: {exc}") from exc
    return config.model_copy(update={"experiment_id": stable_experiment_id(config)}), notes


def _target_is_skewed_positive(task_spec: TaskSpec, profile: DataProfile) -> bool:
    if task_spec.task_type != "regression":
        return False
    ts = profile.target_summary
    try:
        long_tail = ts["max"] > ts["mean"] + 4 * ts["std"] or ts["mean"] > 1.05 * ts["50%"]
        return ts.get("n_nonpositive", 1) == 0 and long_tail
    except (KeyError, TypeError):
        return False


def seed_hypotheses(task_spec: TaskSpec, profile: DataProfile) -> list[Hypothesis]:
    """Strong, cheap defaults run before any LLM idea — they guarantee a good submission early."""
    n_classes = len(task_spec.class_labels or [])
    big_multiclass = task_spec.task_type == "classification" and n_classes > 2 and profile.n_rows_train > 30_000
    seeds = [
        Hypothesis(
            title="LightGBM default",
            rationale="fast, strong baseline for tabular data",
            kind="model",
            base="scratch",
            model_name="lightgbm",
            # multiclass boosting grows one tree per class per round, so fewer rounds there
            hyperparameters={
                "n_estimators": 300 if big_multiclass else 600, "learning_rate": 0.1 if big_multiclass else 0.05,
                "num_leaves": 63, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
                "min_child_samples": 20,
            },
        ),
        Hypothesis(
            title="CatBoost default",
            rationale="strong with categorical features out of the box",
            kind="model",
            base="scratch",
            model_name="catboost",
            hyperparameters=(
                {"iterations": 400, "learning_rate": 0.12, "depth": 6}
                if big_multiclass
                else {"iterations": 1000, "learning_rate": 0.08, "depth": 6}
            ),
        ),
    ]
    return seeds


def playbook_hypotheses(task_spec: TaskSpec, profile: DataProfile, drop_candidates: list[str]) -> list[Hypothesis]:
    """Generic ideas tried in order when there is no LLM (or after it ran out). Applied on top of the best."""
    ideas: list[Hypothesis] = []
    if _target_is_skewed_positive(task_spec, profile):
        ideas.append(Hypothesis(title="log1p target", kind="preprocessing", preprocessing={"target_transform": "log1p"},
                                rationale="right-skewed non-negative target"))
    ideas += [
        Hypothesis(title="LightGBM more trees, lower LR", kind="params", model_name="lightgbm",
                   hyperparameters={"n_estimators": 1500, "learning_rate": 0.03, "num_leaves": 63, "subsample": 0.8,
                                    "subsample_freq": 1, "colsample_bytree": 0.7, "min_child_samples": 20},
                   rationale="lower learning rate with more trees usually generalizes better"),
        Hypothesis(title="LightGBM regularized", kind="params", model_name="lightgbm",
                   hyperparameters={"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.8,
                                    "subsample_freq": 1, "colsample_bytree": 0.6, "min_child_samples": 50,
                                    "reg_lambda": 5.0},
                   rationale="if more capacity hurts, the split rewards simpler models (e.g. a shift between periods)"),
        Hypothesis(title="CatBoost shorter, stronger L2", kind="params", model_name="catboost",
                   hyperparameters={"iterations": 400, "learning_rate": 0.1, "depth": 6, "l2_leaf_reg": 10},
                   rationale="less overfitting than the default CatBoost seed"),
        Hypothesis(title="missing-value indicators", kind="features", add_feature_steps=["missing_indicators"],
                   rationale="missingness itself can be informative"),
        Hypothesis(title="count encoding of categoricals", kind="features", add_feature_steps=["count_encode"],
                   rationale="category frequency is a cheap, leak-free signal"),
        Hypothesis(title="XGBoost", kind="model", model_name="xgboost",
                   hyperparameters={"n_estimators": 800, "learning_rate": 0.05, "max_depth": 8, "subsample": 0.8,
                                    "colsample_bytree": 0.8, "min_child_weight": 2, "max_cat_threshold": 32},
                   rationale="different tree-growing strategy, sometimes better"),
        Hypothesis(title="CatBoost deeper", kind="params", model_name="catboost",
                   hyperparameters={"iterations": 1200, "learning_rate": 0.06, "depth": 8},
                   rationale="more interactions between features"),
        Hypothesis(title="LightGBM bigger leaves", kind="params", model_name="lightgbm",
                   hyperparameters={"n_estimators": 1000, "learning_rate": 0.04, "num_leaves": 127, "subsample": 0.8,
                                    "subsample_freq": 1, "colsample_bytree": 0.7, "min_child_samples": 10},
                   rationale="larger trees for a large dataset"),
    ]
    ideas += [
        Hypothesis(title=f"drop {col}", kind="features", add_feature_steps=[f"drop:{col}"],
                   rationale=f"{col} may be a period/group id that does not transfer to test")
        for col in drop_candidates
    ]
    return ideas
