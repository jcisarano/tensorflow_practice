import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, accuracy_score


def run():
    X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape)

    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X_train, y_train)

    y_pred = tree_model.predict(X_test)
    print(accuracy_score(y_test, y_pred))

    tree_grid_search = DecisionTreeClassifier(random_state=42)
    params = {"max_leaf_nodes": list(range(2, 200)), "min_samples_split": [2, 3, 4]}
    grid_search = GridSearchCV(tree_grid_search, params, verbose=2, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_estimator_, grid_search.best_estimator_.__getattribute__("min_samples_split"))

    y_pred = grid_search.best_estimator_.predict(X_test)
    print(accuracy_score(y_test, y_pred))

    max_leaf_nodes = grid_search.best_estimator_.__getattribute__("max_leaf_nodes")
    min_samples_split = grid_search.best_estimator_.__getattribute__("min_samples_split")

    # Grow a forest
    # data set has 10000 items, so create train set of 100 items with 20% test set
    split = ShuffleSplit(n_splits=1000, train_size=0.0125, test_size=0.0025, random_state=42)

    forest = []
    for train_index, test_index in split.split(X_train):
        print("TRAIN", len(train_index), "TEST", len(test_index))
        tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes,
                                      min_samples_split=min_samples_split,
                                      random_state=42)
        tree.fit(X_train[train_index], y_train[train_index])
        forest.append(tree)

    for tree in forest:
        y_pred = tree.predict(X_test)
        print(accuracy_score(y_test, y_pred))
