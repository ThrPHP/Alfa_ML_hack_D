"""Minimal, no-LLM demonstration of the full ml_core pipeline on the real
multiclass classification task (network attack type, Macro F1).

Run from the project root:

    python examples/run_without_llm_multiclass.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_core import experiment_log, prediction, profiling, training, validation
from ml_core.contracts import ExperimentConfig

DATA_DIR = Path(__file__).resolve().parents[1] / "Base" / "Datasets" / "Многоклассовая классификация"
RUN_DIR = Path(__file__).resolve().parent / "runs" / "multiclass"


def main() -> None:
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    dict_path = DATA_DIR / "data_dictionary.csv"
    sub_path = DATA_DIR / "sample_submission.csv"

    task_spec = profiling.infer_task_spec(train_path, dict_path, sub_path)
    print(f"task_type={task_spec.task_type} metric={task_spec.metric_name} n_classes={len(task_spec.class_labels)}")

    data_profile = profiling.profile_data(train_path, dict_path, test_path)
    print(f"\n{len(data_profile.warnings)} profiling warnings:")
    for w in data_profile.warnings:
        print(f"  - {w}")

    val_plan = validation.create_validation(
        task_spec, train_path, RUN_DIR / "validation", n_splits=5, seed=42, strategy="stratified_kfold",
    )
    print(f"\nvalidation: strategy={val_plan.strategy} n_splits={val_plan.n_splits} warnings={val_plan.warnings}")

    # Safety net first: a valid submission.csv must exist before anything else,
    # because the platform hard-kills the process after 3 hours.
    fallback_path = RUN_DIR / "submission.csv"
    fallback_result = prediction.write_fallback_submission(task_spec, data_profile, train_path, test_path, fallback_path)
    print(f"\nfallback submission -> {fallback_result.status} ({fallback_path})")

    candidates = [
        ExperimentConfig(model_name="baseline", validation_id=val_plan.validation_id),
        ExperimentConfig(
            model_name="catboost",
            validation_id=val_plan.validation_id,
            hyperparameters={"iterations": 300, "depth": 6},
        ),
    ]

    print("\nrunning experiments...")
    for cfg in candidates:
        t0 = time.time()
        result = training.run_experiment(task_spec, cfg, val_plan, data_profile, train_path, RUN_DIR / "experiments")
        print(
            f"  {cfg.model_name:20s} -> {result.status:10s} macro_f1={result.metric_value} "
            f"(+-{result.metric_std}) {time.time()-t0:.1f}s"
        )

    best = experiment_log.select_best(RUN_DIR / "experiments", validation_id=val_plan.validation_id)
    assert best is not None, "no successful, fully-comparable experiment found"
    print(f"\nbest: {best.config.model_name} macro_f1={best.metric_value:.3f}")

    pred_result = prediction.refit_and_predict(task_spec, best.config, data_profile, train_path, test_path, fallback_path)
    print(f"\nrefit_and_predict -> {pred_result.status}, rows={pred_result.n_rows}, checks={pred_result.checks_passed}")
    print(f"submission written to {fallback_path}")


if __name__ == "__main__":
    main()
