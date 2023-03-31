from src.clean import remove_rowid
from src.read import read_xlsx


def write_parquet(df, fname):
    """
    Write a pandas dataframe to a parquet file
    :param df: the dataframe to write to a parquet file
    :param fname: the fname to write to
    :return: None
    """
    df.to_parquet(fname)


if __name__ == '__main__':
    path = 'data/'
    fname = 'existing-customers.xlsx'
    df = read_xlsx(path + fname)
    new_fname = fname.split('.')[0] + '.parquet'
    write_parquet(df, path + new_fname)
    fname = 'potential-customers.xlsx'
    df = read_xlsx(path + fname)
    new_fname = fname.split('.')[0] + '.parquet'
    write_parquet(df, path + new_fname)
