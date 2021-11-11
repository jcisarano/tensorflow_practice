import numpy as np


def get_3d_dataset(m=60, w1=0.1, w2=0.3, noise=0.1):
    np.random.seed(4)

    angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
    X = np.empty((m, 3))
    X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m)/2
    X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m)/2
    X[:, 2] = X[:, 0]*w1 + X[:, 1]*w2 + noise * np.random.randn(m)

    return X