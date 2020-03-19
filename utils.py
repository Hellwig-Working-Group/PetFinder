"""This module contains various utility functions."""

from tabulate import tabulate


def pretty_print(df):
    """Pretty prints Pandas DataFrame"""
    print(tabulate(df, headers='keys', tablefmt='psql'))