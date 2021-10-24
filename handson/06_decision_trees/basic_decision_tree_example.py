import os

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

import visualization as vv


def run():
    iris = load_iris()

    X = iris.data[:, 2:]  # petal length and width
    y = iris.target

    tree_clf = DecisionTreeClassifier(max_depth=2)
    tree_clf.fit(X, y)

    # vv.graphviz_image(tree_clf, iris)

    plt.figure(figsize=(8, 4))
    vv.plot_decision_boundary(tree_clf, X, y, legend=True)
    plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
    plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
    plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)
    plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)
    plt.text(1.40, 1.0, "Depth=0", fontsize=15)
    plt.text(3.2, 1.80, "Depth=1", fontsize=13)
    plt.text(4.05, 0.5, "Depth=2", fontsize=11)
    plt.show()
