import os

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

import visualization as vv

def run():
    iris = load_iris()

    X = iris.data[:, 2:]  # petal length and width
    y = iris.target

    tree_clf = DecisionTreeClassifier(max_depth=2)
    tree_clf.fit(X, y)

    print(tree_clf)

    vv.graphviz_image(tree_clf, iris)