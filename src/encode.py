import pandas as pd
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    pass


def dummify(df):
    """
    Dummify the dataframe
    :param df: the dataframe to dummify
    :return: a dummified dataframe
    """
    return pd.get_dummies(df)


def label_encode(df):
    """
    Label encode the dataframe
    :param df: the dataframe to label encode
    :return: a label encoded dataframe
    """
    return df.apply(LabelEncoder().fit_transform)
