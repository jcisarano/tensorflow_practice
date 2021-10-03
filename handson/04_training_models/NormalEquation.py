import numpy as np


def generate_data(random_seed=42):
    np.random.seed(random_seed)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    return X, y
