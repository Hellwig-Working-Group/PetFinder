"""This module contains various preprocessing utilities."""


def remove_object_cols(df):
    """Removes string columns from dataframe."""
    object_cols = df.dtypes[df.dtypes == 'object'].index.values
    return df.drop(columns=object_cols)
