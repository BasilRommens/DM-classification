import numpy as np


def get_cost_matrix():
    return np.array([
        [0, 0.05 * 310 + 10],
        [0, -0.1 * 980 + 10]
    ])


if __name__ == '__main__':
    pass
