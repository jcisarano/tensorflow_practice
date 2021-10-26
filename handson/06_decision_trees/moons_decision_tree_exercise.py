import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import clone

from scipy.stats import mode


def run():
    X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape)

    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X_train, y_train)

    y_pred = tree_model.predict(X_test)
    print("Single classifier accuracy:", accuracy_score(y_test, y_pred))

    tree_grid_search = DecisionTreeClassifier(random_state=42)
    params = {"max_leaf_nodes": list(range(2, 200)), "min_samples_split": [2, 3, 4]}
    grid_search = GridSearchCV(tree_grid_search, params, verbose=1, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    # print(grid_search.best_estimator_, grid_search.best_estimator_.__getattribute__("min_samples_split"))

    y_pred = grid_search.best_estimator_.predict(X_test)
    print("Best estimator accuracy:", accuracy_score(y_test, y_pred))

    max_leaf_nodes = grid_search.best_estimator_.__getattribute__("max_leaf_nodes")
    min_samples_split = grid_search.best_estimator_.__getattribute__("min_samples_split")

    # Grow a forest
    # data set has 10000 items, so create train set of 100 items with 20% test set
    split = ShuffleSplit(n_splits=1000, train_size=0.0125, test_size=0.0025, random_state=42)

    forest = []
    for train_index, test_index in split.split(X_train):
        tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes,
                                      min_samples_split=min_samples_split,
                                      random_state=42)
        tree.fit(X_train[train_index], y_train[train_index])
        forest.append(tree)

    pred = None
    for tree in forest:
        y_pred = tree.predict(X_test)
        if pred is None:
            pred = y_pred
        else:
            pred = np.c_[pred, y_pred]
        # print(pred.shape)
        # print("Accuracy: ", accuracy_score(y_test, y_pred), " ", y_pred[0], " ", X_test[0])

    y_pred, counts = mode(pred, axis=1)
    # print(y_pred.ravel().shape)
    print("Forest accuracy: ", accuracy_score(y_test, y_pred))
    # print(mode(pred, axis=1))


    # Forest book solution:
    n_trees = 1000
    n_instances = 100
    mini_sets = []

    # creates train size by subtracting desired len from full set size and using what's left
    # i.e. test_size=8000-100, so test_size = total-test_size). Test set is not used
    rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances, random_state=42)
    for mini_train_index, mini_test_index in rs.split(X_train):
        X_mini_train = X_train[mini_train_index]
        y_mini_train = y_train[mini_train_index]
        # print(X_mini_train.shape, y_mini_train.shape)
        mini_sets.append((X_mini_train, y_mini_train))

    # loop to clone best estimator 1000 times
    forest = [clone(grid_search.best_estimator_) for _ in range(n_trees)]
    accuracy_scores = []

    for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
        tree.fit(X_mini_train, y_mini_train)

        y_pred = tree.predict(X_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))

    print("Book mean accuracy:", np.mean(accuracy_scores))

    Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)

    for tree_index, tree in enumerate(forest):
        Y_pred[tree_index] = tree.predict(X_test)

    y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)
    print("Book forest accuracy:", accuracy_score(y_test, y_pred_majority_votes.reshape([-1])))

