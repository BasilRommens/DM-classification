import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin

from src.exploration import show_null_vals
from src.read import read_parquet


# https://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
# 31/03/2022
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """
        Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].mean()
                               for c in X],
                              index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


if __name__ == '__main__':
    path = 'data/'
    fname = 'existing-customers.parquet'
    df = read_parquet(path + fname)
    dft = DataFrameImputer().fit_transform(df)
    show_null_vals(dft)  # no output expected
