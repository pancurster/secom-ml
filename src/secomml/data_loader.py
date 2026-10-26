#!/usr/bin/env python3
"""Load the SECOM dataset and labels from the raw data files."""

from pathlib import Path
import zipfile
import pandas as pd

def load_secom(zip_file_path: str) -> pd.DataFrame:
    '''
    Takes path to secom zip dataset.
    Returns pandas dataframe. Both Labels & Data concatenated
    '''
    file_list = unzip_raw_data(zip_file_path)
    unzip_dir = file_list[0].parent

    secom_data_path = list(unzip_dir.glob('secom.data'))
    assert len(secom_data_path) == 1
    secom_data_df = load_secom_data(secom_data_path[0])

    secom_labels_path = list(unzip_dir.glob('secom_labels.data'))
    assert len(secom_labels_path) == 1
    secom_labels_df = load_secom_labels(secom_labels_path[0])

    secom = pd.concat([
        secom_labels_df,
        secom_data_df
        ],
        axis=1,
        )

    #return secom, secom_labels_df, secom_data_df
    return secom

def unzip_raw_data(zip_file_path: str, destination_path: str = None) -> list:
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

def load_secom_data(data_path) -> pd.DataFrame:
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

def load_secom_labels(labels_path) -> pd.DataFrame:
    '''
    Loads secom labels file
    '''
    df = pd.read_csv(
        labels_path,
        sep=' ',
        header=None,
        names=['pass', 'timestamp'],
        na_values=['NaN'],
        parse_dates=[1],
        date_format='%d/%m/%Y %H:%M:%S',
        cache_dates=True,
        dtype_backend='numpy_nullable',
        )
    return df
