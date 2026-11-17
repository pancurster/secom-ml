#!/usr/bin/env python3
"""Load the SECOM dataset and labels from the raw data files."""

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

def load_secom(zip_file_path: str) -> pd.DataFrame:
    secom = load_raw_secom(zip_file_path)
    secom = convert_column_to_boolean(secom, column_name='faulty')
    secom = convert_to_optimal_integer_type(secom)
    return secom

def convert_to_optimal_integer_type(df):
    """
    Convert integer types in 'df' to its optimal integer size type
    """
    df_copy = df.copy()
    df_int = df_copy.select_dtypes(include=np.integer)
    for col in df_int.columns:
        downcast_type = 'signed' if any(df_int[col] < 0) else 'unsigned'
        df_copy[col] = pd.to_numeric(df_int[col], downcast=downcast_type, dtype_backend='numpy_nullable')
    return df_copy

def convert_column_to_boolean(df, column_name='faulty'):
    """
    Convert column to boolean type.

    Assume it is secom 'faulty' column, so map the values -1 to 0 and 1 kept as 1.
    """
    df_copy = df.copy()
    df_copy[column_name] = df_copy[column_name].map({-1:0, 1:1})
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

def load_raw_secom(zip_file_path: str) -> pd.DataFrame:
    '''
    Takes path to secom zip dataset.
    Returns pandas dataframe. Both Labels & Data concatenated
    '''
    file_list = unzip_raw_data(zip_file_path)
    unzip_dir = file_list[0].parent

    secom_data_path = list(unzip_dir.glob('secom.data'))
    assert len(secom_data_path) == 1
    secom_data_df = load_raw_secom_data(secom_data_path[0])

    secom_labels_path = list(unzip_dir.glob('secom_labels.data'))
    assert len(secom_labels_path) == 1
    secom_labels_df = load_raw_secom_labels(secom_labels_path[0])

    secom = pd.concat([
        secom_labels_df,
        secom_data_df
        ],
        axis=1,
        )
    return secom

def unzip_raw_data(zip_file_path: str, destination_path: str = '') -> list:
    '''
    Unzip archive. Return paths list of extracted files.
    '''
    src = Path(zip_file_path)
    dst = Path(destination_path) if destination_path else src.parent

    if not src.exists():
        print(f'Path {src} does not exists!')
        return []
    if not dst.exists():
        print(f'Path {dst} does not exists!')
        return []

    with zipfile.ZipFile(src) as z:
        z.extractall(dst)
        extracted_files = [dst / Path(f.filename) for f in z.filelist]
        return extracted_files

def load_raw_secom_data(data_path) -> pd.DataFrame:
    '''
    Loads secom data file
    '''
    df = pd.read_csv(
        data_path,
        sep=' ',
        header=None,
        na_values=['NaN'],
        float_precision='high',
        dtype_backend='numpy_nullable',
        ).add_prefix('x')
    return df

def load_raw_secom_labels(labels_path) -> pd.DataFrame:
    '''
    Loads secom labels file
    '''
    df = pd.read_csv(
        labels_path,
        sep=' ',
        header=None,
        names=['faulty', 'timestamp'],
        na_values=['NaN'],
        parse_dates=[1],
        date_format='%d/%m/%Y %H:%M:%S',
        cache_dates=True,
        dtype_backend='numpy_nullable',
        )
    return df
