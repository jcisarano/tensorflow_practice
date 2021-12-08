import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import BayesianGaussianMixture

from visualization_helpers import plot_gaussian_mixture

"""
BayesianGaussianMixture class will find the correct number of clusters by giving weights equal (or close to) zero
for unnecessary clusters. Set the number of components higher than you will need, and it will eliminate unneeded clusters
automatically.
"""


def get_blob_data():
    x1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
    x1 = x1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
    x2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
    x2 = x2 + [6, -8]
    X = np.r_[x1, x2]
    y = np.r_[y1, y2]

    return X, y


def bgm_simple_ex(X, y):
    bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
    bgm.fit(X)

    print(np.round(bgm.weights_, 2))

    plt.figure(figsize=(8, 5))
    plot_gaussian_mixture(bgm, X)
    plt.show()


def run():
    X, y = get_blob_data()
    bgm_simple_ex(X, y)