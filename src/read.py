import pandas as pd


def read_xlsx(fname):
    """
    Read xlsx file and return a pandas dataframe
    :param fname: the fname to read from
    :return: a pandas dataframe
    """
    return pd.read_excel(fname)


def read_parquet(fname):
    """
    Read parquet file and return a pandas dataframe
    :param fname: the fname to read from
    :return: a pandas dataframe
    """
    return pd.read_parquet(fname)


if __name__ == '__main__':
    path = 'data/'
    fname = 'existing-customers.xlsx'
    df = read_xlsx(path + fname)
    print(df)
