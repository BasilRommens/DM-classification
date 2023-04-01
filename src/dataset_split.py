from sklearn.model_selection import StratifiedKFold, train_test_split

from src.read import read_parquet


def get_X_y(df):
    y = df['class']
    X = df.drop('class', axis=1)

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
    return train_test_split(X, y, test_size=0.2, stratify=df['class'])


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
