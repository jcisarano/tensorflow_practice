import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.mixture import BayesianGaussianMixture

from visualization_helpers import plot_gaussian_mixture, plot_data

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


def bgm_low_v_high(X, y):
    bgm_low = BayesianGaussianMixture(n_components=10, max_iter=1000, n_init=1,
                                      weight_concentration_prior=0.01, random_state=42)
    bgm_high = BayesianGaussianMixture(n_components=10, max_iter=1000, n_init=1,
                                       weight_concentration_prior=10000, random_state=42)
    nn = 73
    bgm_low.fit(X[:nn])
    bgm_high.fit(X[:nn])

    print(np.round(bgm_low.weights_, 2))
    print(np.round(bgm_high.weights_, 2))

    plt.figure(figsize=(9, 4))

    plt.subplot(121)
    plot_gaussian_mixture(bgm_low, X[:nn])
    plt.title("weight_concentration_prior = {}".format(bgm_low.weight_concentration_prior), fontsize=14)

    plt.subplot(122)
    plot_gaussian_mixture(bgm_high, X[:nn], show_ylabels=False)
    plt.title("weight_concentration_prior = {}".format(bgm_high.weight_concentration_prior), fontsize=14)

    plt.show()


def bgm_moons():
    X_moons, y_moons = make_moons(n_samples=1000, noise=0.05, random_state=42)
    bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
    bgm.fit(X_moons)

    plt.figure(figsize=(9, 3.2))

    plt.subplot(121)
    plot_data(X_moons)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)

    plt.subplot(122)
    plot_gaussian_mixture(bgm, X_moons, show_ylabels=False)

    plt.show()

def run():
    # X, y = get_blob_data()
    # bgm_simple_ex(X, y)
    # bgm_low_v_high(X, y)
    bgm_moons()

    