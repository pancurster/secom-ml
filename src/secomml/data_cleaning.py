"""Contains functions for data cleaning and preprocessing."""
import numpy as np
import pandas as pd

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

