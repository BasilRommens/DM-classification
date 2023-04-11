import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.clean import remove_rowid
from src.dataset_split import split_train_test
from src.encode import dummify
from src.exploration import show_null_vals
from src.models.decision_tree_classifier import decision_tree
from src.models.gradient_boosted_trees_classifier import gradient_boosted_trees
from src.read import read_parquet


def impute_decision_tree(X):
    # set the final dataframe
    final_df = X.copy()

    cols = find_null_cols(X)

    # set all the null values to -1
    X[cols] = X[cols].fillna(-1)

    for col in cols:
        test_df, train_df = train_test_null(X, col)

        # split the training and test data into X and y
        test_X, test_y, train_X, train_y = split_train_test(col, test_df,
                                                            train_df)

        # label encode the target variable
        le = LabelEncoder()
        train_y = le.fit_transform(train_y)

        # dummify the test and training data
        # combine both the training and test data X
        _X = pd.concat([train_X, test_X])
        _X = dummify(_X)

        # split the combined data into training and test data
        train_X = _X.loc[train_X.index]
        test_X = _X.loc[test_X.index]

        # train a decision tree model
        clf = decision_tree(train_X, train_y)
        pred_y = clf.predict(test_X)

        # convert the values to the original data type
        pred_y = le.inverse_transform(pred_y)

        # set the values in the final dataframe
        final_df.loc[test_df.index, col] = pred_y

    return final_df


def train_test_null(X, col):
    test_df = X[X[col] == -1]  # test where values are missing
    train_df = X[X[col] != -1]  # train where values are not missing
    return test_df, train_df


def find_null_cols(X):
    # find the columns in which the values are missing
    cols = list()
    for col in X.columns:
        if X[col].isnull().any():
            cols.append(col)
    return cols


if __name__ == '__main__':
    path = 'data/'
    fname = 'existing-customers.parquet'
    df = read_parquet(path + fname)
    df = remove_rowid(df)
    dft = impute_decision_tree(df)

    show_null_vals(dft)  # no output expected
