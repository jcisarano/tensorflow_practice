# utility functions
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

PROJECT_ROOT_DIR = "."
IMAGE_SAVE_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGE_SAVE_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def load_iris_setosa_or_versicolor():
    iris = load_iris()
    X = iris["data"][:, (2, 3)]
    y = iris["target"]

    setosa_or_versicolor = (y == 0) | (y == 1)
    X = X[setosa_or_versicolor]
    y = y[setosa_or_versicolor]

    return X, y


def load_iris_data_width_length(iris_type=2):
    """
    :param iris_type: flower type to select for, 0 = setosa, 1 = versicolor, 2 = virginica
    :return: petal width/length data, with given type==1, all others 0
    """
    iris = load_iris()
    X = iris["data"][:, (2, 3)]
    y = (iris["target"] == iris_type).astype(np.float64)

    return X, y

def plot_svc_decision_boundary(svm_clf, xmin, xmax, show_support_vectors=True, show_gutters=True):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # at the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    if show_support_vectors:
        svs = svm_clf.support_vectors_
        plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors="#FFAAAA")
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    if show_gutters:
        plt.plot(x0, gutter_up, "k--", linewidth=2)
        plt.plot(x0, gutter_down, "k--", linewidth=2)


def plot_dataset(X, y, axes, show=True):
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
    plt.axis(axes)
    plt.grid(True, which="both")
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
    if show:
        plt.show()


def plot_predictions(clf, axes, show=True):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)
    if show:
        plt.show()


