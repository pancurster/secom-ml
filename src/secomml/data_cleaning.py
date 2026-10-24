def drop_zero_variance_columns(df):
    """
    Drop columns with zero variance from the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A new DataFrame with zero variance columns dropped.
    List: A list of column names that were dropped.
    """
    df_copy = df.copy()
    zero_variance_columns = df_copy.loc[:, df_copy.var() == 0].columns
    df_copy.drop(columns=zero_variance_columns, inplace=True)
    return df_copy, zero_variance_columns
