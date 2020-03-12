"""This module contains utils for data loading."""

from os import listdir
from os.path import join
from collections import defaultdict
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import DATA_DIR


def get_label_dataframes(data_dir=DATA_DIR):
    """Returns a dict with filenames as keys and Pandas Dataframes as values."""
    label_files = [file for file in listdir(data_dir) if file.endswith('csv')]
    return {filename: pd.read_csv(join(DATA_DIR, filename)) for filename in label_files}


def get_json_data(data_type, data_dir=DATA_DIR):
    """
    Returns a nested dict with filenames as keys and dicts from json as values
    :param data_dir: root data directory
    :param data_type: either 'metadata' or 'sentiment', this keyword decides which json data is pulled
    """
    assert data_type in ('sentiment', 'metadata')
    metadata_dirs = [directory for directory in listdir(data_dir) if data_type in directory]
    data = defaultdict(lambda: dict())
    for directory in metadata_dirs:
        dir_path = join(data_dir, directory)
        files = listdir(dir_path)
        for file in files:
            assert file.endswith('json'), 'Wrong files in metadata directory.'
            filepath = join(dir_path, file)
            with open(filepath, 'r') as f:
                data[directory][file] = json.load(f)
    return dict(data)


def get_dataset(dataset_type, data_dir=DATA_DIR):
    """
    Returns Pandas Dataframe with selected dataset type from csv
    :param dataset_type: either test or train
    :param data_dir: root data directory
    :return:
    """
    assert dataset_type in ('train', 'test')
    filepath_suffix = join(dataset_type, f'{dataset_type}.csv')
    return pd.read_csv(join(data_dir, filepath_suffix))


# todo: decide if we want to use raw image data
def get_images():
    raise NotImplementedError


# def create_train_and_test(df, target_col, test_size=0.3, seed=997):
#     """Creates train/test split from dataframe"""
#     assert target_col in df.columns, f"Wrong target col: {target_col}"
#     y = df[target_col]
#     x = df.drop(columns=[target_col])
#     return train_test_split(x, y, test_size=test_size, random_state=seed)


def remove_object_cols(df):
    """Removes string columns from dataframe."""
    object_cols = df.dtypes[df.dtypes == 'object'].index.values
    return df.drop(columns=object_cols)
