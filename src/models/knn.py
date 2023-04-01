from sklearn.neighbors import KNeighborsClassifier

from src.clean import remove_rowid, change_class, dummify
from src.dataset_split import get_stratified_split
from src.impute import DataFrameImputer
from src.read import read_parquet


def knn(train_X, train_y, test_X, test_y, k):
    # train a model
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(train_X, train_y)

    # evaluate the model
    score = knn.score(test_X, test_y)
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

    # dummify the dataframe
    df = dummify(df)

    # create a train and test set
    train_X, test_X, train_y, test_y = get_stratified_split(df)

    # apply the knn model
    for k in range(1, 20):
        knn(train_X, train_y, test_X, test_y, k=k)
