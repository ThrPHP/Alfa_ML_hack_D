import pydantic
import pytest

from ml_core.contracts import ExperimentConfig, PredictionRequirements, TaskSpec


def test_classification_requires_class_labels():
    with pytest.raises(pydantic.ValidationError):
        TaskSpec(
            task_type="classification",
            target_column="y",
            metric_name="macro_f1",
            greater_is_better=True,
            prediction_requirements=PredictionRequirements(output_columns=["id", "y"]),
        )


def test_unknown_field_rejected():
    with pytest.raises(pydantic.ValidationError):
        ExperimentConfig(model_name="catboost", validation_id="x", not_a_real_field=1)


def test_experiment_result_round_trips_through_json():
    from ml_core.contracts import ExperimentResult

    cfg = ExperimentConfig(model_name="catboost", validation_id="v1")
    result = ExperimentResult(
        status="success",
        experiment_id="e1",
        metric_name="rmse",
        greater_is_better=False,
        metric_value=1.23,
        validation_id="v1",
        n_folds_completed=5,
        n_folds_requested=5,
        config=cfg,
    )
    restored = ExperimentResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.is_fully_comparable is True
