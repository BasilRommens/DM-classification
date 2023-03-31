from src.read import read_parquet


def show_null_vals(df):
    """
    Show the number of null values in each column
    :param df: the dataframe to show the null values from
    :return: None
    """
    for col in df.columns:
        n_null_vals = df[col].isnull().sum()
        if n_null_vals > 0:
            print(f'{col} has {n_null_vals} null values')


if __name__ == '__main__':
    path = 'data/'
    fname = 'existing-customers.parquet'
    df = read_parquet(path + fname)

    # show the count of the null values present in the dataframe
    show_null_vals(df)
