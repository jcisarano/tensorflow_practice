# first examples of SVM, comparison to standard linear regression models, show sensitivity to feature scales

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC

import utils


def plot_bad_models(X, y):
    """
    the plot on the left shows several linear regression models, none of which does a great job with this data.
    the plot on the right is an SVM, and the solid line is the decision boundary between the classes. The dashed lines
    are the support vectors.
    :param X:
    :param y:
    :return:
    """
    svm_clf = SVC(kernel="linear", C=float("inf"))
    svm_clf.fit(X, y)
    x0 = np.linspace(0, 5.5, 200)
    pred_1 = 5 * x0 - 20
    pred_2 = x0 - 1.8
    pred_3 = 0.1 * x0 + 0.5

    fig, axes = plt.subplots(ncols=2, figsize=(10, 2.7), sharey=True)
    plt.sca(axes[0])
    plt.plot(x0, pred_1, "g--", linewidth=2)
    plt.plot(x0, pred_2, "m--", linewidth=2)
    plt.plot(x0, pred_3, "r-", linewidth=2)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", label="Iris versicolor")
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", label="Iris setosa")
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="upper left", fontsize=14)
    plt.axis([0, 5.5, 0, 2])

    plt.sca(axes[1])
    utils.plot_svc_decision_boundary(svm_clf=svm_clf, xmin=0, xmax=5.5)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bo")
    plt.xlabel("Petal length", fontsize=14)
    plt.axis([0, 5.5, 0, 2])
    plt.show()


def plot_sensitivity_to_scales():
    """
    Shows SVM sensitivity to feature scaling.
    The left plot is unscaled, and the margins are narrow, but wide.
    The right plot is scaled and the decision boundaries have much more room.
    :return:
    """
    Xs = np.array([[1, 50], [5, 20], [3, 80], [5, 60]]).astype(np.float64)
    ys = np.array([0, 0, 1, 1])
    svm_clf = SVC(kernel="linear", C=100)
    svm_clf.fit(Xs, ys)

    plt.figure(figsize=(9, 2.7))
    plt.subplot(121)
    plt.plot(Xs[:, 0][ys == 1], Xs[:, 1][ys == 1], "bo")
    plt.plot(Xs[:, 0][ys == 0], Xs[:, 1][ys == 0], "ms")
    utils.plot_svc_decision_boundary(svm_clf=svm_clf, xmin=0, xmax=6)
    plt.xlabel("$x_0$", fontsize=20)
    plt.xlabel("$x_1$", fontsize=20, rotation=0)
    plt.title("Unscaled", fontsize=16)
    plt.axis([0, 6, 0, 90])

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xs)
    svm_clf.fit(X_scaled, ys)

    plt.subplot(122)
    plt.plot(X_scaled[:, 0][ys == 1], X_scaled[:, 1][ys == 1], "bo")
    plt.plot(X_scaled[:, 0][ys == 0], X_scaled[:, 1][ys == 0], "ms")
    utils.plot_svc_decision_boundary(svm_clf=svm_clf, xmin=-2, xmax=2)
    plt.xlabel("$x'_0$", fontsize=20)
    plt.ylabel("$x'_1$", fontsize=20, rotation=0)
    plt.title("Scaled", fontsize=16)
    plt.axis([-2, 2, -2, 2])
    plt.show()


def run():
    iris = load_iris()
    X = iris["data"][:, (2, 3)]
    y = iris["target"]

    setosa_or_versicolor = (y == 0) | (y == 1)
    X = X[setosa_or_versicolor]
    y = y[setosa_or_versicolor]

    plot_bad_models(X, y)
    plot_sensitivity_to_scales()
