#!/usr/bin/env python3
"""Utility functions for Exploratory Data Analysis (EDA) for the SECOM dataset."""
import pandas as pd

def profile(dataframe, max_examples=3):
    """Return a summary table for each column in the dataframe."""
    n = len(dataframe)
    rows = []
    for column in dataframe.columns:
        s = dataframe[column]
        non_null = s.notna().sum() # number of values that are not NaN
        rekord = {
            "column": column,
            "type": str(s.dtype),
            "non_null": non_null,
            "missing": s.isna().sum(),
            "missing_%": s.isna().sum() / len(s) * 100,
            "unique": s.nunique(),
            "unique_%": s.nunique() / len(s) * 100,
            "examples": ", ".join(map(str, s.dropna().unique()[:max_examples])),
        }
        if pd.api.types.is_numeric_dtype(s) and non_null > 0:
            rekord["min"] = round(float(s.min()), 2)
            rekord["median"] = s.median()
            rekord["max"] = s.max()
        rows.append(rekord)
    return pd.DataFrame(rows).set_index("column")