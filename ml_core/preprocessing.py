"""Leakage-safe preprocessing.

Two modes, selected by ExperimentConfig.preprocessing.mode:

- "native": for tree ensembles (catboost/lightgbm/xgboost/hist_gradient_boosting).
  Numeric NaNs are left as NaN (every one of those libraries splits on
  missingness natively); categorical NaNs and genuinely unseen categories
  (confirmed present in this competition's data: `state` has values in
  test.csv absent from train.csv) are mapped to an explicit "__missing__"
  token rather than erroring. No encoder is *fit* on the target.

- "linear": for models that need a fully numeric, dense matrix (Ridge,
  LogisticRegression). Median/most-frequent imputation, one-hot encoding
  for low-cardinality categoricals (handle_unknown="ignore"), and a small
  custom FrequencyEncoder for high-cardinality ones — picked specifically
  over target encoding, which the task brief allows deferring and which
  needs extra out-of-fold machinery to stay leak-safe.

Every fit_* function in this module must only ever be called with the
training fold; training.py enforces that by construction (fit happens
inside the per-fold loop, transform is then applied to the validation fold).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_core.contracts import ColumnProfile, PreprocessingConfig

MISSING_TOKEN = "__missing__"


@dataclass
class FeatureColumns:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    low_card_categorical: list[str] = field(default_factory=list)
    high_card_categorical: list[str] = field(default_factory=list)


def split_feature_columns(
    columns: list[ColumnProfile], config: PreprocessingConfig
) -> FeatureColumns:
    out = FeatureColumns()
    for c in columns:
        if c.role != "feature":
            continue
        if config.drop_constant_columns and c.inferred_kind == "constant":
            continue
        if c.inferred_kind in ("numeric", "binary"):
            out.numeric.append(c.name)
        else:
            out.categorical.append(c.name)
            if c.n_unique > config.high_cardinality_threshold:
                out.high_card_categorical.append(c.name)
            else:
                out.low_card_categorical.append(c.name)
    return out


# --------------------------------------------------------------------------- #
# Native mode (tree models)
# --------------------------------------------------------------------------- #


class CategoryVocabulary:
    """Fixed category sets derived ONLY from the fold given to `.fit`.

    Values not seen during `.fit` become NaN on `.transform` — all four
    supported tree libraries treat a NaN/missing category as a legitimate
    split direction, so this is a safe, standard way to handle categories
    that show up for the first time in validation/test data.
    """

    def __init__(self) -> None:
        self.categories_: dict[str, list[str]] = {}

    def fit(self, df: pd.DataFrame, categorical_cols: list[str]) -> "CategoryVocabulary":
        for col in categorical_cols:
            filled = df[col].astype("object").where(df[col].notna(), MISSING_TOKEN).astype(str)
            self.categories_[col] = sorted(filled.unique().tolist())
        return self

    def transform(self, df: pd.DataFrame, categorical_cols: list[str], as_category_dtype: bool) -> pd.DataFrame:
        out = df.copy()
        for col in categorical_cols:
            filled = out[col].astype("object").where(out[col].notna(), MISSING_TOKEN).astype(str)
            if as_category_dtype:
                out[col] = pd.Categorical(filled, categories=self.categories_[col])
            else:
                # CatBoost wants plain strings; unseen values are kept as-is
                # (CatBoost itself handles novel categorical values at predict time).
                out[col] = filled
        return out


def prepare_native_frame(
    df: pd.DataFrame,
    feature_cols: FeatureColumns,
    vocab: CategoryVocabulary,
    model_name: str,
) -> pd.DataFrame:
    """Select feature columns and apply the fitted CategoryVocabulary.

    `model_name == "catboost"` keeps categoricals as plain strings (CatBoost's
    own cat_features handling); every other native-mode model gets pandas
    `category` dtype, which lightgbm/xgboost/HistGradientBoosting all read
    natively.
    """
    cols = feature_cols.numeric + feature_cols.categorical
    as_category_dtype = model_name != "catboost"
    out = vocab.transform(df[cols], feature_cols.categorical, as_category_dtype=as_category_dtype)
    return out


# --------------------------------------------------------------------------- #
# Linear mode
# --------------------------------------------------------------------------- #


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Maps each category to its relative frequency in the fitted data.

    Frequency is a property of the column's own values, not of the target,
    so fitting it per-CV-fold (as training.py does) cannot leak target
    information — only the usual "we don't know the true population
    frequency, only the fold's" approximation, same as any other fold-fit
    statistic.
    """

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        self.freq_maps_ = {
            col: X[col].astype(str).fillna(MISSING_TOKEN).value_counts(normalize=True).to_dict()
            for col in X.columns
        }
        self.columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        out = np.zeros((len(X), len(self.columns_)), dtype=float)
        for i, col in enumerate(self.columns_):
            freq_map = self.freq_maps_[col]
            out[:, i] = X[col].astype(str).fillna(MISSING_TOKEN).map(freq_map).fillna(0.0).to_numpy()
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_freq" for c in self.columns_])


def build_linear_preprocessor(feature_cols: FeatureColumns, config: PreprocessingConfig) -> ColumnTransformer:
    numeric_steps = []
    if config.numeric_imputer != "none":
        numeric_steps.append(("impute", SimpleImputer(strategy=config.numeric_imputer)))
    if config.scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps) if numeric_steps else "passthrough"

    low_card_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_TOKEN)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    high_card_pipeline = Pipeline([("freq", FrequencyEncoder())])

    transformers = []
    if feature_cols.numeric:
        transformers.append(("numeric", numeric_pipeline, feature_cols.numeric))
    if feature_cols.low_card_categorical:
        transformers.append(("low_card", low_card_pipeline, feature_cols.low_card_categorical))
    if feature_cols.high_card_categorical:
        transformers.append(("high_card", high_card_pipeline, feature_cols.high_card_categorical))

    return ColumnTransformer(transformers, remainder="drop")


# --------------------------------------------------------------------------- #
# Target transform
# --------------------------------------------------------------------------- #


def forward_target_transform(y: np.ndarray, kind: str) -> np.ndarray:
    if kind == "none":
        return y
    if kind == "log1p":
        return np.log1p(y)
    raise ValueError(f"Unknown target_transform {kind!r}")


def inverse_target_transform(y: np.ndarray, kind: str) -> np.ndarray:
    if kind == "none":
        return y
    if kind == "log1p":
        return np.expm1(y)
    raise ValueError(f"Unknown target_transform {kind!r}")
