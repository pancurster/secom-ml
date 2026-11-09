"""Contains functions for data cleaning and preprocessing."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import validate_data

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

class MaxNanFeatureSelector(SelectorMixin, BaseEstimator):
    def __init__(self, *, max_nan=0.9):
        self.max_nan = max_nan

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def fit(self, X, y=None):
        Xcolumns = X.columns if type(X) == pd.DataFrame else []
        X = validate_data(self, X, reset=True, dtype=None, ensure_all_finite="allow-nan")
        self.support_mask_ = np.asarray(pd.isna(X).mean(axis=0) < self.max_nan)
        self.removed_mask_ = ~self.support_mask_
        self.feature_names_removed_ = Xcolumns[self.removed_mask_] if len(Xcolumns) else 'NA'
        return self

    def _get_support_mask(self):
        return self.support_mask_

class MaxConstFeatureSelector(SelectorMixin, BaseEstimator):
    def __init__(self, *, max_const_ratio=0.98, ref_const=None):
        self.max_const_ratio = max_const_ratio
        self.ref_const = ref_const

    def fit(self, X, y=None):
        X_cols = X.columns if type(X) == pd.DataFrame else []
        X_n = X.shape[0]
        X = validate_data(self, X, reset=True, dtype=None)
        X_consts = np.median(X, axis=0) if self.ref_const == None else self.ref_const
        self.const_ratio_ = 1 - (np.count_nonzero(X - X_consts, axis=0) / X_n)
        self.support_mask_ = self.const_ratio_ < self.max_const_ratio
        self.removed_mask_ = ~self.support_mask_
        self.feature_names_removed_ = X_cols[self.removed_mask_] if len(X_cols) else 'NA'
        return self

    def _get_support_mask(self):
        return self.support_mask_

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

def non_zero_variance_columns(df):
    """
    Get column names with NON zero variance
    """
    nzv_names = pd.Series([col for col in df.columns if df[col].var() != 0])
    return nzv_names

def zero_variance_columns(df):
    """
    Get column names with zero variance
    """
    zv_names = pd.Series([col for col in df.columns if df[col].var() == 0])
    return zv_names
