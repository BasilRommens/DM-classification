import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.read import read_parquet


def get_X_y(df: pd.DataFrame, target: str = 'class'):
    y = df[target]
    X = df.drop(target, axis=1)

    return X, y


def get_stratified_kfold_split(df, k=5):
    X, y = get_X_y(df)

    # StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # split the samples
    split = skf.split(X, y)

    return split


def get_stratified_split(df):
    X, y = get_X_y(df)
    return train_test_split(X, y, test_size=0.2, stratify=df['class'],
                            random_state=45)


if __name__ == '__main__':
    path = 'data/'
    fname = 'existing-customers.parquet'
    df = read_parquet(path + fname)

    split = get_stratified_kfold_split(df)
    for i, (train_index, test_index) in enumerate(split):
        print(f'Fold {i}')
        print(f'\tTrain index: {train_index}')
        print(f'\tTest index: {test_index}')

    print(get_stratified_split(df))


def split_train_test(col, test_df, train_df):
    train_X, train_y = get_X_y(train_df, col)
    test_X, test_y = get_X_y(test_df, col)
    return test_X, test_y, train_X, train_y
