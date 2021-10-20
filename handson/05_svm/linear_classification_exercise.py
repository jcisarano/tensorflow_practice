# Exercise:
# Train a LinearSVC on a linearly separable dataset.
# Then train an SVC and a SGDClassifier on the same dataset.
# See if you can get them to produce roughly the same model.
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import utils


def run():
    # from iris data, use setosa and versicolor because they are linearly separable
    iris = load_iris()
    X = iris["data"][:, (2, 3)]  # only petal length and width
    y = iris["target"]

    setosa_or_versicolor = (y == 0) | (y == 1)
    X = X[setosa_or_versicolor]
    y = y[setosa_or_versicolor]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    linear_svc_clf = LinearSVC(C=5, loss="hinge", random_state=42)
    linear_svc_clf.fit(X, y)



    plt.figure(figsize=(10, 7))
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^", label="Iris setosa")
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs", label="Iris versicolor")
    utils.plot_svc_decision_boundary(linear_svc_clf, xmin=0, xmax=6, show_support_vectors=False, show_gutters=False)
    plt.xlabel("Petal width", fontsize=16)
    plt.ylabel("Petal length", fontsize=16)
    plt.legend(loc="upper left")
    plt.show()


