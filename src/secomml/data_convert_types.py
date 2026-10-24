def convert_pass_to_boolean(df, column_name='pass'):
    """
    Convert a specified column in the DataFrame to boolean type.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column to convert.

    Returns:
    pd.DataFrame: A new DataFrame with the specified column converted to boolean type.
    """
    df_copy = df.copy()
    df_copy[column_name] = df_copy[column_name].map(lambda x: 1 if x == -1 else 0)
    df_copy[column_name] = df_copy[column_name].astype(pd.BooleanDtype())
    return df_copy
def convert_to_integer_type(df, column_names, target_dtype):
    """
    Convert specified columns in the DataFrame to a given integer type.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_names (list): List of column names to convert.
    target_dtype (str): The target integer type (e.g., 'Int8', 'Int16', 'Int32', 'Int64').

    Returns:
    pd.DataFrame: A new DataFrame with the specified columns converted to the target integer type.
    """
    df_copy = df.copy()
    for col in column_names:
        df_copy[col] = df_copy[col].astype(target_dtype)
    return df_copy
