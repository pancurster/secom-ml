"""Contains functions for data cleaning and preprocessing."""
import pandas as pd

def drop_zero_variance_columns(df, return_droped=False):
    """
    Drop columns with zero variance from the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A new DataFrame with zero variance columns dropped.
    List: A list of column names that were dropped.
    """
    df_copy = df.copy()
    xnames = df_copy.columns.str.startswith("x")
    xvar = df_copy.loc[:, xnames]
    zero_variance_columns = xvar.loc[:, xvar.var() == 0].columns
    if return_droped:
        return df_copy.drop(columns=zero_variance_columns), zero_variance_columns.tolist()
    return df_copy.drop(columns=zero_variance_columns)
