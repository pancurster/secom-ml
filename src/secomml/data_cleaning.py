"""Contains functions for data cleaning and preprocessing."""
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

