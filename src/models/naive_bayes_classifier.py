from sklearn.naive_bayes import CategoricalNB

from src.clean import change_class, remove_rowid
from src.dataset_split import get_stratified_split
from src.encode import label_encode, dummify
from src.impute import DataFrameImputer
from src.read import read_parquet


def naive_bayes(train_X, train_y, test_X, test_y):
    # train a model
    clf = CategoricalNB()
    clf.fit(train_X, train_y)

    # evaluate the model
    score = clf.score(test_X, test_y)
    print(f'Accuracy: {score}')
    return score


if __name__ == '__main__':
    path = 'data/'
    fname = 'existing-customers.parquet'

    # read the dataset
    df = read_parquet(path + fname)

    # remove the rowid column
    df = remove_rowid(df)

    # impute the missing values
    df = DataFrameImputer().fit_transform(df)

    # change the class column to a binary column
    df = change_class(df)

    # label encode the dataframe
    df = label_encode(df)

    # create a train and test set
    train_X, test_X, train_y, test_y = get_stratified_split(df)

    # apply the naive bayes model
    naive_bayes(train_X, train_y, test_X, test_y)
