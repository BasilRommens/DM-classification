from sklearn.model_selection import StratifiedKFold

from src.read import read_parquet


def get_split(df):
    y = df['class']
    X = df.drop('class', axis=1)

    # StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # split the samples
    split = skf.split(X, y)

    return split


if __name__ == '__main__':
    path = 'data/'
    fname = 'existing-customers.parquet'
    df = read_parquet(path + fname)

    split = get_split(df)
    for i, (train_index, test_index) in enumerate(split):
        print(f'Fold {i}')
        print(f'\tTrain index: {train_index}')
        print(f'\tTest index: {test_index}')
