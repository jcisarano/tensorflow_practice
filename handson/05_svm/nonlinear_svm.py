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
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42, max_iter=10000)),
    ])
    clf.fit(X, y)

    return clf


def nonlinear_svm_w_polynomial_kernel(X, y, coef0=1, degree=3):
    """
    Example of kernel trick. Uses polynomial kernel, which can be faster than adding high-degree polynomials
    It doesn't actually add polynomials? just mimics the same effect???
    degree var controls number of polynomials
    coef0 determines how much the model is influenced by high-degree polynomials
    :param X:
    :param y:
    :param coef0:
    :param degree:
    :return:
    """
    from sklearn.svm import SVC
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=degree, coef0=coef0, C=5))
    ])
    clf.fit(X, y)
    return clf


def plot_poly_kernel_clfs(poly_clf, poly100_clf, X, y):
    fig, axes = plt.subplots(ncols=2, figsize=(10.5, 4.5), sharey=True)
    plt.sca(axes[0])
    utils.plot_predictions(clf=poly_clf, axes=[-1.5, 2.45, -1, 1.5], show=False)
    utils.plot_dataset(X=X, y=y, axes=[-1.5, 2.4, -1, 1.5], show=False)
    plt.title(r"$d=3, r=1, c=5$", fontsize=18)

    plt.sca(axes[1])
    utils.plot_predictions(clf=poly100_clf, axes=[-1.5, 2.45, -1, 1.5], show=False)
    utils.plot_dataset(X=X, y=y, axes=[-1.5, 2.4, -1, 1.5], show=False)
    plt.title(r"$d=10, r=100, C=5$", fontsize=18)
    plt.ylabel("")
    plt.show()


def run():
    # adding_features_to_dataset()
    X, y = get_and_plot_moon_dataset()
    # polynomial_clf = nonlinear_svm_w_polynomial_features(X, y)
    # utils.plot_predictions(clf=polynomial_clf, axes=[-1.5, 2.5, -1, 1.5], show=False)
    # utils.plot_dataset(X=X, y=y, axes=[-1.5, 2.5, -1, 1.5])

    poly_kernel_svm_clf = nonlinear_svm_w_polynomial_kernel(X=X, y=y)
    poly100_kernel_svm_clf = nonlinear_svm_w_polynomial_kernel(X=X, y=y, coef0=100, degree=10)
    plot_poly_kernel_clfs(poly_clf=poly_kernel_svm_clf, poly100_clf=poly100_kernel_svm_clf, X=X, y=y)


