from sklearn.impute import KNNImputer
import pandas as pd

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
    # before imputing the null values, we need to convert the categorical
    # columns to numerical columns through one-hot encoding
    df = pd.get_dummies(df, dummy_na=True)
    for col in df.columns:
        if 'workclass' in col:
            print(col)
            print(df[col].isnull().sum())

    imputer = KNNImputer(n_neighbors=2)
    df = imputer.fit_transform(df)
    df = pd.DataFrame(df)
    print(df)
