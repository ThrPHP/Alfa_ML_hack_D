"""Minimal, no-LLM demonstration of the full ml_core pipeline on the real
regression task (building energy use, RMSE).

Run from the project root:

    python examples/run_without_llm_regression.py

It profiles the data, builds two validation strategies to show *why* the
choice matters for this specific dataset (plain k-fold vs. a time-aware
split — see README.md "Why time_split for the regression task"), runs a
small set of candidate configs, and writes a final submission.csv next to
this script using refit_and_predict().
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_core import experiment_log, prediction, profiling, training, validation
from ml_core.contracts import ExperimentConfig, PreprocessingConfig

DATA_DIR = Path(__file__).resolve().parents[1] / "Base" / "Datasets" / "Регрессия"
RUN_DIR = Path(__file__).resolve().parent / "runs" / "regression"


def main() -> None:
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    dict_path = DATA_DIR / "data_dictionary.csv"
    sub_path = DATA_DIR / "sample_submission.csv"

    task_spec = profiling.infer_task_spec(train_path, dict_path, sub_path)
    print(f"task_type={task_spec.task_type} metric={task_spec.metric_name} notes={task_spec.notes}")

    data_profile = profiling.profile_data(train_path, dict_path, test_path)
    print(f"\n{len(data_profile.warnings)} profiling warnings:")
    for w in data_profile.warnings:
        print(f"  - {w}")

    # Year_Factor has 5 distinct values in train (1..5) and test is entirely a 6th,
    # unseen value — a single held-out-latest-period split mirrors that structure
    # (time_split needs n_splits+1 distinct train values, so n_splits=1 is the
    # most this column supports).
    val_plan = validation.create_validation(
        task_spec, train_path, RUN_DIR / "validation", n_splits=1, seed=42, strategy="time_split",
        time_column="Year_Factor",
    )
    print(f"\nvalidation: strategy={val_plan.strategy} n_splits={val_plan.n_splits} warnings={val_plan.warnings}")

    candidates = [
        ExperimentConfig(model_name="baseline", validation_id=val_plan.validation_id),
        ExperimentConfig(
            model_name="ridge",
            validation_id=val_plan.validation_id,
            preprocessing=PreprocessingConfig(mode="linear", scale_numeric=True),
        ),
        ExperimentConfig(model_name="hist_gradient_boosting", validation_id=val_plan.validation_id),
        ExperimentConfig(
            model_name="catboost",
            validation_id=val_plan.validation_id,
            hyperparameters={"iterations": 400, "depth": 6, "learning_rate": 0.1},
        ),
        ExperimentConfig(
            model_name="catboost",
            validation_id=val_plan.validation_id,
            hyperparameters={"iterations": 400, "depth": 6, "learning_rate": 0.1},
            preprocessing=PreprocessingConfig(mode="native", target_transform="log1p"),
        ),
    ]

    print("\nrunning experiments...")
    for cfg in candidates:
        t0 = time.time()
        result = training.run_experiment(task_spec, cfg, val_plan, data_profile, train_path, RUN_DIR / "experiments")
        tag = f"{cfg.model_name}" + (" +log1p" if cfg.preprocessing.target_transform == "log1p" else "")
        print(f"  {tag:30s} -> {result.status:10s} rmse={result.metric_value} (+-{result.metric_std}) {time.time()-t0:.1f}s")

    best = experiment_log.select_best(RUN_DIR / "experiments", validation_id=val_plan.validation_id)
    assert best is not None, "no successful, fully-comparable experiment found"
    print(f"\nbest: {best.config.model_name} rmse={best.metric_value:.3f}")

    out_path = RUN_DIR / "submission.csv"
    print(f"\nrefitting {best.config.model_name} on the full train set and predicting test (prints nothing while it runs)...")
    t0 = time.time()
    pred_result = prediction.refit_and_predict(task_spec, best.config, data_profile, train_path, test_path, out_path)
    print(f"refit_and_predict finished in {time.time()-t0:.1f}s")
    print(f"\nrefit_and_predict -> {pred_result.status}, rows={pred_result.n_rows}, checks={pred_result.checks_passed}")
    print(f"submission written to {out_path}")


if __name__ == "__main__":
    main()
