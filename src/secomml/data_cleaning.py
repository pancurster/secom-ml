"""Contains functions for data cleaning and preprocessing."""
from numbers import Real

import numpy as np
import pandas as pd

from scipy.stats import fisher_exact
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import validate_data
from sklearn.utils._param_validation import Interval

# TODO del
def drop_zero_variance_columns(df, return_droped=False):
    """
    Drop columns with zero variance from the DataFrame.
    """
    df_copy = df.copy()
    xnames = df_copy.columns.str.startswith("x")
    xvar = df_copy.loc[:, xnames]
    zero_variance_columns = xvar.loc[:, xvar.var() == 0].columns
    if return_droped:
        return df_copy.drop(columns=zero_variance_columns), zero_variance_columns.tolist()
    return df_copy.drop(columns=zero_variance_columns)

class MaxNanThreshold(SelectorMixin, BaseEstimator):

    _parameter_constraints: dict = {
        "threshold": [Interval(Real, 0, 1, closed="both")]
    }

    def __init__(self, *, threshold=0.0):
        self.threshold = threshold

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def fit(self, X, y=None):
        X_cols = X.columns if type(X) == pd.DataFrame else []
        X = validate_data(self, X, reset=True, dtype=None, ensure_all_finite="allow-nan")
        self.removed_mask_ = np.asarray(pd.isna(X).mean(axis=0) >= self.threshold)
        self.support_mask_ = ~self.removed_mask_
        self.feature_names_removed_ = X_cols[self.removed_mask_] if len(X_cols) else 'NA'
        return self

    def _get_support_mask(self):
        return self.support_mask_

class MaxConstThreshold(SelectorMixin, BaseEstimator):

    _parameter_constraints: dict = {
        "threshold": [Interval(Real, 0, 1, closed="both")]
    }

    def __init__(self, *, threshold=0.98):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_names = X.columns if type(X) == pd.DataFrame else []
        X = validate_data(self, X, reset=True, dtype=None)
        n_samples = X.shape[0]
        if y is not None:
            y = np.asarray(y)
            if y.shape[0] != n_samples:
                raise ValueError("y must have the same number of rows as X")

        # Most frequent value of each feature as the reference constant.
        reference_values = np.empty(X.shape[1], dtype=X.dtype)
        for idx in range(X.shape[1]):
            column = X[:, idx]
            values, counts = np.unique(column, return_counts=True)
            reference_values[idx] = values[np.argmax(counts)]

        # 1) shift by const toward 0, 2) count not 0, 3) ratio c/n, 4) const-ratio = (1 - ratio_non_zero)
        self.const_ratio_ = 1 - (np.count_nonzero(X - reference_values, axis=0) / n_samples)
        candidate_mask = self.const_ratio_ >= self.threshold

        if y is not None:
            # Fisher test: not-const can possibly correlate with `y`
            not_const_mask = ~np.isclose(X, reference_values)
            for idx in np.where(candidate_mask)[0]:
                ct = pd.crosstab(not_const_mask[:, idx], y)
                _, p_value = fisher_exact(ct)
                if p_value <= 0.0005:
                    candidate_mask[idx] = False

        self.removed_mask_ = candidate_mask
        self.support_mask_ = ~candidate_mask
        self.feature_names_removed_ = X_names[self.removed_mask_] if len(X_names) else 'NA'
        return self

    def _get_support_mask(self):
        return self.support_mask_

# TODO del
def drop_zero_percentage_columns(df, ths=0.98, na_policy='as_zero'):
    """
    Drop columns with proportion of `0.0` values more than `ths` (0-1)
    """
    assert ths > 0 and ths < 1
    assert na_policy == 'as_zero' or na_policy == 'as_value'

    df_copy = df.copy()
    eps = np.finfo(np.float64).eps
    if (na_policy=='as_zero'):
        zp = (( ((df < eps) & (df > -eps)) | (df.isna())).sum() / df.shape[0])
    elif (na_policy=='as_value'):
        zp = (( ((df < eps) & (df > -eps))).sum() / df.shape[0])
    else:
        assert 'wrong na_policy value'
    print(zp)
    zp_names = zp.index[(zp >= ths)]
    return df_copy.drop(columns=zp_names)
# TODO del
def non_zero_variance_columns(df):
    """
    Get column names with NON zero variance
    """
    nzv_names = pd.Series([col for col in df.columns if df[col].var() != 0])
    return nzv_names
# TODO del
def zero_variance_columns(df):
    """
    Get column names with zero variance
    """
    zv_names = pd.Series([col for col in df.columns if df[col].var() == 0])
    return zv_names
