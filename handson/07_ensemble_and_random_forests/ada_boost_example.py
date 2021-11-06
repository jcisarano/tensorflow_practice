"""
    Boosting refers to any ensemble method that combines several weak learners into one strong learner.
    Learners are trained sequentially, each one trying to improve upon the previous one.
    AdaBoost and Gradient Boosting are the most popular methods
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

import data_utils as du


def plot_adaboost(X_train, y_train):
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5, random_state=42
    )
    ada_clf.fit(X_train, y_train)
    X, y = du.get_raw_moons()
    du.plot_decision_boundary(ada_clf, X, y)
    plt.show()


def run():
    X_train, X_test, y_train, y_test = du.get_moons()
    plot_adaboost(X_train=X_train, y_train=y_train)
