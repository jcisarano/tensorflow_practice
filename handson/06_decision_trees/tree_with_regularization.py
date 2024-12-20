import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

import visualization as vv


def plt_seeded_tree_example(X, y):
    """
    decision trees use stochastic algorithm, so using different random seed creates a different tree
    :return:
    """
    tree_clf_seeded = DecisionTreeClassifier(max_depth=2, random_state=40)
    tree_clf_seeded.fit(X, y)

    plt.figure(figsize=(8, 4))
    vv.plot_decision_boundary(tree_clf_seeded, X, y, legend=False)
    plt.plot([0, 7.5], [0.8, 0.8], "k-", linewidth=2)
    plt.plot([0, 7.5], [1.75, 1.75], "k--", linewidth=2)
    plt.text(1.0, 0.9, "Depth=0", fontsize=15)
    plt.text(1.0, 1.8, "Depth=1", fontsize=13)
    plt.show()


def plt_min_samples_leaf_example():
    from sklearn.datasets import make_moons
    Xm, ym = make_moons(n_samples=100, noise=0.25, random_state=53)

    deep_tree_clf1 = DecisionTreeClassifier(random_state=42)
    deep_tree_clf2 = DecisionTreeClassifier(random_state=42, min_samples_leaf=4)
    deep_tree_clf1.fit(Xm, ym)
    deep_tree_clf2.fit(Xm, ym)

    fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
    plt.sca(axes[0])
    vv.plot_decision_boundary(deep_tree_clf1, Xm, ym, axes=[-1.5, 2.4, -1, 1.5], iris=False)
    plt.title("No restrictions", fontsize=16)
    plt.sca(axes[1])
    vv.plot_decision_boundary(deep_tree_clf2, Xm, ym, axes=[-1.5, 2.4, -1, 1.5], iris=False)
    plt.title("min_samples_leaf = {}".format(deep_tree_clf2.min_samples_leaf), fontsize=14)
    plt.ylabel("")
    plt.show()


def plt_rotated_iris(X, y):
    angle = np.pi / 180 * 20
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    Xr = X.dot(rotation_matrix)

    tree_clrf_rotated = DecisionTreeClassifier(random_state=42)
    tree_clrf_rotated.fit(Xr, y)

    plt.figure(figsize=(8, 3))
    vv.plot_decision_boundary(tree_clrf_rotated, Xr, y, axes=[0.5, 7.5, -1.0, 1], iris=False)
    plt.show()


def plt_rotated_linear():
    np.random.seed(6)
    Xs = np.random.rand(100, 2) - 0.5
    ys = (Xs[:, 0] > 0).astype(np.float32) * 2

    angle = np.pi / 4
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    Xsr = Xs.dot(rotation_matrix)

    tree_clf_s = DecisionTreeClassifier(random_state=42)
    tree_clf_s.fit(Xs, ys)
    tree_clf_sr = DecisionTreeClassifier(random_state=42)
    tree_clf_sr.fit(Xsr, ys)

    fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
    plt.sca(axes[0])
    vv.plot_decision_boundary(tree_clf_s, Xs, ys, axes=[-0.7, 0.7, -0.7, 0.7], iris=False)
    plt.sca(axes[1])
    vv.plot_decision_boundary(tree_clf_sr, Xsr, ys, axes=[-0.7, 0.7, -0.7, 0.7], iris=False)
    plt.ylabel("")
    plt.show()


def run():
    iris = load_iris()

    X = iris.data[:, 2:]  # petal length and width
    y = iris.target

    # plt_seeded_tree_example(X=X, y=y)
    # plt_min_samples_leaf_example()
    # plt_rotated_iris(X, y)
    plt_rotated_linear()
