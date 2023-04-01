from src.impute import DataFrameImputer
from src.read import read_parquet


def remove_rowid(df):
    """
    Remove the rowid column from the dataframe
    :param df: the dataframe to remove the rowid from
    :return: a dataframe without the rowid
    """
    return df.drop('RowID', axis=1)


def change_class(df):
    """
    Change the class column to a binary column, where 0 is <50k and 1 is >=50k
    :param df: the dataframe to change the class column in
    :return: the changed dataframe
    """
    df['class'] = df['class'].apply(lambda x: 0 if x == '<=50K' else 1)

    return df


if __name__ == '__main__':
    path = 'data/'
    fname = 'existing-customers.parquet'
    df = read_parquet(path + fname)
    df = remove_rowid(df)
    df = DataFrameImputer().fit_transform(df)
