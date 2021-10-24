import os

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from graphviz import Source
from sklearn.tree import export_graphviz


def run():
    iris = load_iris()

    X = iris.data[:, 2:]  # petal length and width
    y = iris.target

    tree_clf = DecisionTreeClassifier(max_depth=2)
    tree_clf.fit(X, y)

    print(tree_clf)

    export_graphviz(
        tree_clf,
        out_file=os.path.join("./images/", "iris_tree.dot"),
        feature_names=iris.feature_names[2:],
        rounded=True,
        filled=True
    )
    graph = Source.from_file(os.path.join("./images/", "iris_tree.dot"))
    graph.format = "png"
    graph.render("iris_tree_render", view=True)