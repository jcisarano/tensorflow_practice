"""
Principal Component Analysis (PCA) - popular method of dimensionality reduction that identifies the hyperplane closest
to the data and projects the data onto that plane.

Choosing the correct hyperplane is important to reducing data loss and variance. PCA attempts to do this by reducing
the mean squared distance between the original dataset and its projection onto the new axis.
"""

import numpy as np
from sklearn.decomposition import PCA

import data_utils as du

def pca_manual_example(X):
    # obtain principal components c1, c2 using Singular Value Decomposition (SVD)
    X_centered = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X_centered)
    c1 = Vt.T[:, 0]
    c2 = Vt.T[:, 1]

    # check the data
    m, n = X.shape
    S = np.zeros(X_centered.shape)
    S[:n, :n] = np.diag(s)
    print(np.allclose(X_centered, U.dot(S).dot(Vt)))

    # project down to D dimensions (see formula p. 222)
    W2 = Vt.T[:, :2]
    X2D = X_centered.dot(W2)

    X3D_inv = X2D.dot(Vt[:2, :])

    evr = np.square(s) / np.square(s).sum()
    print("svd explained variance ratio:", evr)
    return X2D, X3D_inv


def pca_using_sklearn(X):
    pca = PCA(n_components=2)
    X2D = pca.fit_transform(X)
    X3D_inv = pca.inverse_transform(X2D)

    print("pca.explained_variance_ratio_:", pca.explained_variance_ratio_)
    print("pca data loss:", 1-pca.explained_variance_ratio_.sum())
    return X2D, X3D_inv, pca.mean_

def run():
    X = du.get_3d_dataset()
    print(X.shape)
    X2D_using_svd, X3D_inv_using_svd = pca_manual_example(X)
    print(X2D_using_svd.shape)

    X2D, X3D_inv, pca_mean = pca_using_sklearn(X)

    # the results are the same, but flipped on both axes:
    print(X2D[:5])
    print(X2D_using_svd[:5])

    # flip the axes back, and they are the same:
    print(np.allclose(X2D, -X2D_using_svd))

    # Compare the original to the reconstructed inverse.
    # They are close, but not the same:
    print(np.allclose(X3D_inv, X))

    # Compute the error between original and reconstructed data:
    print(np.mean(np.sum(np.square(X3D_inv - X), axis=1)))

    # after correcting sklearn version for mean, results are the same:
    print(np.allclose(X3D_inv_using_svd, X3D_inv - pca_mean))



