from sklearn.ensemble import RandomForestClassifier


def random_forest(train_X, train_y):
    # train a model
    clf = RandomForestClassifier()
    clf.fit(train_X, train_y)

    return clf


if __name__ == '__main__':
    from src.clean import change_class, remove_rowid
    from src.dataset_split import get_stratified_split
    from src.encode import dummify
    from src.impute import impute_decision_tree
    from src.read import read_parquet

    path = 'data/'
    fname = 'existing-customers.parquet'

    # read the dataset
    df = read_parquet(path + fname)

    # remove the rowid column
    df = remove_rowid(df)

    # impute the missing values
    df = impute_decision_tree(df)

    # change the class column to a binary column
    df = change_class(df)

    # label encode the dataframe
    df = dummify(df)

    # create a train and test set
    train_X, test_X, train_y, test_y = get_stratified_split(df)

    # apply the naive bayes model
    random_forest(train_X, train_y)
