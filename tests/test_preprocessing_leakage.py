import numpy as np
import pandas as pd

from ml_core.preprocessing import CategoryVocabulary, FrequencyEncoder, MISSING_TOKEN


def test_frequency_encoder_is_fit_only_on_the_data_it_is_given():
    """Simulates what training.run_experiment does: fit on the training fold
    only, transform the validation fold. The validation fold's own value
    counts must have zero influence on the encoding."""
    fold_train = pd.DataFrame({"cat": ["x"] * 9 + ["y"] * 1})  # x: 90%, y: 10% in-fold
    fold_val = pd.DataFrame({"cat": ["y"] * 20})  # all "y": if this leaked in, "y" freq would be ~1.0

    enc = FrequencyEncoder().fit(fold_train)
    val_transformed = enc.transform(fold_val)
    assert np.allclose(val_transformed[:, 0], 0.1)  # learned from train fold (1/10), not val fold


def test_frequency_encoder_handles_unseen_category_without_error():
    fold_train = pd.DataFrame({"cat": ["x", "x", "y"]})
    fold_val = pd.DataFrame({"cat": ["z", "z"]})  # never seen during fit
    enc = FrequencyEncoder().fit(fold_train)
    out = enc.transform(fold_val)
    assert np.array_equal(out[:, 0], [0.0, 0.0])


def test_category_vocabulary_fit_only_sees_fold_given():
    fold_train = pd.DataFrame({"state": ["A", "B", "A"]})
    vocab = CategoryVocabulary().fit(fold_train, ["state"])
    assert set(vocab.categories_["state"]) == {"A", "B"}


def test_category_vocabulary_unseen_value_becomes_nan_not_an_error():
    fold_train = pd.DataFrame({"state": ["A", "B", "A"]})
    fold_val = pd.DataFrame({"state": ["A", "UNSEEN_AT_TRAIN_TIME", None]})
    vocab = CategoryVocabulary().fit(fold_train, ["state"])

    transformed = vocab.transform(fold_val, ["state"], as_category_dtype=True)
    assert transformed["state"].dtype.name == "category"
    assert pd.isna(transformed["state"].iloc[1])  # unseen category -> missing, not a crash
    assert pd.isna(transformed["state"].iloc[2])  # true NaN stays missing


def test_category_vocabulary_catboost_mode_fills_missing_token_as_string():
    fold_train = pd.DataFrame({"state": ["A", None, "A"]})
    vocab = CategoryVocabulary().fit(fold_train, ["state"])
    transformed = vocab.transform(fold_train, ["state"], as_category_dtype=False)
    assert transformed["state"].dtype == object
    assert (transformed["state"] == MISSING_TOKEN).sum() == 1
