import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, \
    ConfusionMatrixDisplay, confusion_matrix

from src.clean import remove_rowid, change_class
from src.cost_matrix import get_cost_matrix
from src.dataset_split import get_stratified_split, get_stratified_kfold_split, \
    get_X_y
from src.encode import dummify
from src.impute import DataFrameImputer
from src.models.decision_tree_classifier import decision_tree
from src.models.gradient_boosted_trees import gradient_boosted_trees
from src.models.knn_classifier import knn
from src.models.naive_bayes_classifier import naive_bayes
from src.models.random_forest_classifier import random_forest
from src.read import read_parquet
import matplotlib.pyplot as plt


def _confusion_matrix(clf, X, y):
    disp = ConfusionMatrixDisplay.from_estimator(
        clf, X, y,
        display_labels=['<50k', '>=50k'],
        cmap=plt.cm.Blues,
    )
    disp.ax_.set_title(f'{clf.__class__.__name__}')

    plt.show()


def evaluate(model, X, y):
    score = model.score(X, y)
    print(f'Accuracy: {score}')

    # get the prediction of the model
    y_pred = model.predict(X)

    # plot the confusion matrix
    _confusion_matrix(model, X, y)

    # calculate the precision
    precision = precision_score(y, y_pred)
    print(f'Precision: {precision}')

    # calculate the recall
    recall = recall_score(y, y_pred)
    print(f'Recall: {recall}')

    # calculate the f1 score
    f1 = f1_score(y, y_pred)
    print(f'F1: {f1}')

    # get the confusion matrix
    cm = confusion_matrix(y, y_pred)
    cost = (cm * get_cost_matrix()).sum()
    print('Cost: ', cost)

    return score, precision, recall, f1, cost


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
    true_train_X, true_test_X, true_train_y, true_test_y = get_stratified_split(df)
    df = pd.concat([true_train_X, true_train_y], axis=1)
    split = get_stratified_kfold_split(df)

    models = {'decision tree': list(),
              'gradient boosted trees': list(),
              'knn classifier': list(),
              'naive bayes': list(),
              'random forest': list()}
    for train_idces, test_idces in split:
        train_df = df.iloc[train_idces]
        test_df = df.iloc[test_idces]
        train_X, train_y = get_X_y(train_df)
        test_X, test_y = get_X_y(test_df)

        print('decision tree')
        model = decision_tree(train_X, train_y)
        _, _, _, _, cost = evaluate(model, test_X, test_y)
        models['decision tree'].append(cost)

        print()
        print('gradient boosted trees')
        model = gradient_boosted_trees(train_X, train_y)
        _, _, _, _, cost = evaluate(model, test_X, test_y)
        models['gradient boosted trees'].append(cost)

        print()
        print('knn classifier')
        model = knn(train_X, train_y, 5)
        _, _, _, _, cost = evaluate(model, test_X, test_y)
        models['knn classifier'].append(cost)

        # print()
        # print('naive bayes')
        # model = naive_bayes(train_X, train_y)
        # evaluate(model, test_X, test_y)
        # _, _, _, _, cost = evaluate(model, test_X, test_y)
        # models['naive bayes'].append(cost)

        print()
        print('random forest')
        model = random_forest(train_X, train_y)
        evaluate(model, test_X, test_y)
        _, _, _, _, cost = evaluate(model, test_X, test_y)
        models['random forest'].append(cost)

    for model, costs in models.items():
        print(model)
        print(np.average(costs))
        print()

    model = gradient_boosted_trees(true_train_X, true_train_y)
    evaluate(model, true_test_X, true_test_y)
