from sklearn.impute import KNNImputer
import pandas as pd

from src.impute import DataFrameImputer
from src.read import read_parquet


def remove_rowid(df):
    """
    Remove the rowid column from the dataframe
    :param df: the dataframe to remove the rowid from
    :return: a dataframe without the rowid
    """
    return df.drop('RowID', axis=1)


if __name__ == '__main__':
    path = 'data/'
    fname = 'existing-customers.parquet'
    df = read_parquet(path + fname)
    df = remove_rowid(df)
    df = DataFrameImputer().fit_transform(df)
