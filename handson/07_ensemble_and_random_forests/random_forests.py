import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import data_utils as du


def random_forest_v_bag(X_train, y_train, X_test, y_test):
    """
    Example shows that random forest is the same thing as bag of trees
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
    rnd_clf.fit(X_train, y_train)
    y_pred_rf = rnd_clf.predict(X_test)

    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16),
        n_estimators=500, random_state=42
    )
    bag_clf.fit(X_train, y_train)
    y_pred_bag = bag_clf.predict(X_test)
    print(np.sum(y_pred_rf == y_pred_bag) / len(y_pred_bag))


def display_feature_importance():
    """
    Random forest can calculate relative importance of various features
    This function prints percent importance of each feature
    :return:
    """
    iris = load_iris()
    rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
    rnd_clf.fit(iris["data"], iris["target"])
    for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
        print(name, score)


def run():
    # X_train, X_test, y_train, y_test = du.get_moons()
    # random_forest_v_bag(X_train, y_train, X_test, y_test)
    display_feature_importance()
