import matplotlib.pyplot as plt
import numpy as np
# from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline

import utils


def adding_features_to_dataset():
    """
    add features to the dataset to make it linearly classifiable
    the plot on the left is not linearly separable
    so the plot on the right adds one feature to the same data to make it so
    :return:
    """
    X1D = np.linspace(-4, 4, 9).reshape(-1, 1)
    X2D = np.c_[X1D, X1D**2]
    y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])

    plt.figure(figsize=(10, 3))

    plt.subplot(121)
    plt.grid(True, which="both")
    plt.axhline(y=0, color="k")
    plt.plot(X1D[:, 0][y == 0], np.zeros(4), "bs")
    plt.plot(X1D[:, 0][y == 1], np.zeros(5), "g^")
    plt.gca().get_yaxis().set_ticks([])
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.axis([-4.5, 4.5, -0.2, 0.2])

    plt.subplot(122)
    plt.grid(True, which="both")
    plt.axhline(y=0, color="k")
    plt.axvline(x=0, color="k")
    plt.plot(X2D[:, 0][y == 0], X2D[:, 1][y == 0], "bs")
    plt.plot(X2D[:, 0][y == 1], X2D[:, 1][y == 1], "g^")
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
    plt.gca().get_yaxis().set_ticks([0, 4, 8, 12, 16])
    plt.plot([-4.5, 4.5], [6.5, 6.5], "r--", linewidth=3)
    plt.axis([-4.5, 4.5, -1, 17])

    # plt.subplots_adjust(right=1)
    plt.show()


def get_and_plot_moon_dataset(do_plot=True):
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
    if do_plot:
        utils.plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])

    return X, y


def nonlinear_svm_w_polynomial_features(X, y):
    from sklearn.preprocessing import PolynomialFeatures
    clf = Pipeline([
        ("poly_features", PolynomialFeatures()),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42, max_iter=10000)),
    ])
    clf.fit(X, y)

    return clf


def run():
    adding_features_to_dataset()
    X, y = get_and_plot_moon_dataset()
    polynomial_clf = nonlinear_svm_w_polynomial_features(X, y)
