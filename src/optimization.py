from functools import lru_cache

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.clean import remove_rowid, change_class
from src.dataset_split import get_stratified_kfold_split, get_stratified_split, \
    get_X_y
from src.encode import dummify
from src.evaluation import evaluate
from src.impute import impute_decision_tree
from src.read import read_parquet


def get_neighbor(state: tuple) -> tuple:
    neighbor = list(state)
    for i in range(len(state)):
        if type(neighbor[i]) == int:
            neighbor[i] += np.random.randint(-2, 2)
            neighbor[i] = max(0, neighbor[i])  # ensure positive value
        elif type(neighbor[i]) == float:
            neighbor[i] += np.random.uniform(-.1, .1)
            neighbor[i] = np.clip(neighbor[i], 0, 1)  # ensure value in [0, 1]

    return tuple(neighbor)


def get_model_from_state(state: tuple) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(learning_rate=state[0], max_depth=state[1],
                               num_leaves=state[2], feature_fraction=state[3],
                               subsample=state[4], is_unbalance=True)
    return model


@lru_cache(maxsize=3)
def get_cost(state: tuple, split):
    costs = list()

    # get the train and test set indices
    for fold, (train_idces, test_idces) in enumerate(split):
        train_df = df.iloc[train_idces]
        test_df = df.iloc[test_idces]
        train_X, train_y = get_X_y(train_df)
        test_X, test_y = get_X_y(test_df)

        # fit the model
        model = get_model_from_state(state)
        model.fit(train_X, train_y)

        _, _, _, _, cost = evaluate(model, test_X, test_y, verbose=False)
        costs.append(cost)

    return np.mean(costs)


def simulated_annealing(initial_state, split) -> tuple:
    initial_temp = 90
    final_temp = .1
    alpha = 0.95
    current_temp = initial_temp
    current_state = initial_state
    solution = current_state
    best_cost = get_cost(solution, split)
    best_solution = solution

    while current_temp > final_temp:
        neighbor = get_neighbor(current_state)
        current_state_cost = get_cost(current_state, split)
        neighbor_cost = get_cost(neighbor, split)
        cost_diff = current_state_cost - neighbor_cost
        # if the current state cost is lower than the best state cost, update
        # the best state and its cost
        if current_state_cost < best_cost:
            best_solution = current_state
            best_cost = current_state_cost

        if cost_diff > 0:
            current_state = neighbor
        else:
            if np.random.uniform(0, 1) < np.math.exp(cost_diff / current_temp):
                current_state = neighbor

        current_temp *= alpha

    return best_solution


if __name__ == '__main__':
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

    # dummify the dataframe
    df = dummify(df)

    # create a train and test set
    true_train_X, true_test_X, true_train_y, true_test_y = get_stratified_split(
        df)
    df = pd.concat([true_train_X, true_train_y], axis=1)
    split = list(get_stratified_kfold_split(df))

    state = simulated_annealing(initial_state=(.4, 15, 20, .8, .2), split=split)
    print(state)
    model = get_model_from_state(state)
    model.fit(true_train_X, true_train_y)
    evaluate(model, true_test_X, true_test_y)
