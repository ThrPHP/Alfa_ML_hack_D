import numpy as np
import pytest

from ml_core.errors import ValidationMismatchError, ValidationPlanError
from ml_core.profiling import infer_task_spec
from ml_core.validation import create_validation, load_fold_indices


def test_same_seed_gives_identical_folds(regression_dataset, tmp_path):
    paths = regression_dataset
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"])
    plan_a = create_validation(task_spec, paths["train"], tmp_path / "val_a", n_splits=4, seed=7)
    plan_b = create_validation(task_spec, paths["train"], tmp_path / "val_b", n_splits=4, seed=7)
    folds_a = load_fold_indices(plan_a, paths["train"])
    folds_b = load_fold_indices(plan_b, paths["train"])
    assert plan_a.validation_id == plan_b.validation_id
    for fa, fb in zip(folds_a, folds_b):
        assert np.array_equal(np.sort(fa), np.sort(fb))


def test_folds_are_disjoint_and_cover_all_rows(regression_dataset, tmp_path):
    paths = regression_dataset
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"])
    plan = create_validation(task_spec, paths["train"], tmp_path / "val", n_splits=5, seed=1, strategy="kfold")
    folds = load_fold_indices(plan, paths["train"])
    all_idx = np.concatenate(folds)
    assert len(all_idx) == len(set(all_idx.tolist()))  # disjoint
    assert set(all_idx.tolist()) == set(range(300))  # covers every row (300 = regression_dataset n_train)


def test_stratified_kfold_caps_n_splits_to_rarest_class(classification_dataset, tmp_path):
    paths = classification_dataset
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"])
    # ask for way more splits than the rarest class can support
    plan = create_validation(
        task_spec, paths["train"], tmp_path / "val", n_splits=1000, seed=1, strategy="stratified_kfold"
    )
    assert plan.n_splits < 1000
    assert any("reduced" in w for w in plan.warnings)


def test_reusing_plan_against_changed_train_file_raises(regression_dataset, tmp_path):
    paths = regression_dataset
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"])
    plan = create_validation(task_spec, paths["train"], tmp_path / "val", n_splits=4, seed=1)
    # mutate train.csv after the plan was built
    with open(paths["train"], "a", encoding="utf-8") as fh:
        fh.write("99999,1.0,1,Commercial,42.0\n")
    with pytest.raises(ValidationMismatchError):
        load_fold_indices(plan, paths["train"])


def test_time_split_requires_enough_distinct_periods(regression_dataset, tmp_path):
    paths = regression_dataset
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"])
    with pytest.raises(ValidationPlanError):
        create_validation(
            task_spec,
            paths["train"],
            tmp_path / "val",
            strategy="time_split",
            time_column="quality_flag",  # only 2 distinct values, asking for 5 splits
            n_splits=5,
        )
