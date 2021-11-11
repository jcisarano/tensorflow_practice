"""
Principal Component Analysis (PCA) - popular method of dimensionality reduction that identifies the hyperplane closest
to the data and projects the data onto that plane.

Choosing the correct hyperplane is important to reducing data loss and variance. PCA attempts to do this by reducing
the mean squared distance between the original dataset and its projection onto the new axis.
"""

import numpy as np

import data_utils as du

def pca_manual_example(X):
    # obtain principal components c1, c2 using Singular Value Decomposition (SVD)
    X_centered = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X_centered)
    c1 = Vt.T[:, 0]
    c2 = Vt.T[:, 1]

    m, n = X.shape
    S = np.zeros(X_centered.shape)
    S[:n, :n] = np.diag(s)
    print(np.allclose(X_centered, U.dot(S).dot(Vt)))

    # project down to D dimensions
    W2 = Vt.T[:, :2]
    X2D = X_centered.dot(W2)
    return X2D


def run():
    X = du.get_3d_dataset()
    print(X.shape)
    X2D_using_svd = pca_manual_example(X)
    print(X2D_using_svd.shape)


