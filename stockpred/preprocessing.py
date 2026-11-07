"""Leakage-safe preprocessing transformers (sklearn-pipeline compatible).

Everything sample-dependent is learned in ``fit`` from the training sample only:
missingness-based feature selection, imputation fallbacks, winsorisation bounds,
scaling, categorical imputation and encoding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _month_keys(X: pd.DataFrame, date_col: str = "date") -> np.ndarray:
    """Month label of each row, taken from the 'date' index level or column."""
    if isinstance(X.index, pd.MultiIndex) and date_col in X.index.names:
        d = X.index.get_level_values(date_col)
    elif date_col in X.columns:
        d = pd.DatetimeIndex(X[date_col])
    else:
        raise ValueError(f"Input needs a '{date_col}' index level or column")
    d = pd.DatetimeIndex(d)
    return (d.year * 12 + d.month).to_numpy()


class MissingnessFilter(BaseEstimator, TransformerMixin):
    """Drop features whose *training* missing share exceeds ``max_missing``."""

    def __init__(self, max_missing=0.9):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.missing_share_ = X.isna().mean().to_numpy()
        self.keep_mask_ = self.missing_share_ <= self.max_missing
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.loc[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_[self.keep_mask_]


class MonthlyCrossSectionalMeanImputer(BaseEstimator, TransformerMixin):
    """Fill missing predictors with the mean of the *same month's* cross-section.

    A row's signals for return month d were all measured in month d-1, so the
    cross-section mean uses only information available before d's return begins;
    no other month (past or future) enters the mean.

    * ``fit`` stores a training-sample fallback mean per feature.  If a feature
      is observed in no stock in some month (at transform time), that month is
      filled with the training fallback - never with a statistic from the
      validation/test data.
    * Features that are entirely missing in the training sample have no usable
      fallback and are **dropped** from the output (documented deterministic
      policy; reflected in ``get_feature_names_out``).
    """

    def __init__(self, features_to_impute=None, date_col="date"):
        self.features_to_impute = features_to_impute
        self.date_col = date_col

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        _month_keys(X, self.date_col)  # validates date availability
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        if self.features_to_impute is None:
            cols = X.select_dtypes(include=[np.number]).columns
        else:
            cols = [c for c in self.features_to_impute if c in X.columns]
        self.fallback_ = X[list(cols)].mean(skipna=True)  # NaN if all-missing in train
        self.dropped_features_ = list(self.fallback_.index[self.fallback_.isna()])
        self.fallback_ = self.fallback_.dropna()
        self.imputed_features_ = list(self.fallback_.index)
        self.output_features_ = [c for c in X.columns if c not in set(self.dropped_features_)]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        keys = _month_keys(X, self.date_col)
        out = X[self.output_features_].copy()
        if self.imputed_features_:
            sub = out[self.imputed_features_]
            monthly = sub.groupby(keys).transform("mean")  # NaN where a month has no observation
            sub = sub.where(sub.notna(), monthly)
            out[self.imputed_features_] = sub.fillna(self.fallback_[self.imputed_features_])
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.output_features_, dtype=object)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to training-sample quantile bounds; returns float64 ndarray."""

    def __init__(self, lower=0.0025, upper=0.9975):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        A = np.asarray(X)
        self.n_features_in_ = A.shape[1]
        self.feature_names_in_ = (np.asarray(X.columns, dtype=object) if hasattr(X, "columns")
                                  else np.arange(A.shape[1]).astype(str).astype(object))
        qs = np.nanquantile(A.astype(np.float64, copy=False), [self.lower, self.upper], axis=0)
        self.lower_bounds_, self.upper_bounds_ = qs[0], qs[1]
        return self

    def transform(self, X):
        A = np.asarray(X, dtype=np.float64)
        return np.clip(A, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else self.feature_names_in_,
                          dtype=object)


def make_preprocessor(continuous_cols, discrete_cols, max_missing=0.9,
                      winsor=(0.0025, 0.9975)) -> ColumnTransformer:
    """A *fresh, unfitted* preprocessing pipeline.  Call once per fit."""
    numer = make_pipeline(
        MissingnessFilter(max_missing),
        MonthlyCrossSectionalMeanImputer(),
        Winsorizer(*winsor),
        StandardScaler(),
    )
    cat = make_pipeline(
        MissingnessFilter(max_missing),
        SimpleImputer(strategy="most_frequent"),  # drops all-missing columns (fit on train)
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    return ColumnTransformer(
        [("numbers", numer, list(continuous_cols)), ("categoricals", cat, list(discrete_cols))],
        remainder="drop",
        verbose_feature_names_out=True,
    )
