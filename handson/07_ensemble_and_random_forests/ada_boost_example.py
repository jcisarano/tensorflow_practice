"""
    Boosting refers to any ensemble method that combines several weak learners into one strong learner.
    Learners are trained sequentially, each one trying to improve upon the previous one.
    AdaBoost and Gradient Boosting are the most popular methods
"""
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

import data_utils as du


def plot_adaboost(X, y, X_train, y_train):
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5, random_state=42
    )
    ada_clf.fit(X_train, y_train)
    du.plot_decision_boundary(ada_clf, X, y)
    plt.show()


def plot_consecutive(X, y, X_train, y_train):
    """
    Manual example of AdaBoost using SVM classifier.
    Sample weights are weighted on each iteration based on success/failure and used in the subsequent pass.
    Also shows two different learning rates side by side.
    Formulas are shown on pp 201 & 202 of Hands On textbook.
    :param X:
    :param y:
    :param X_train:
    :param y_train:
    :return:
    """
    m = len(X_train)

    _, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
    for subplot, learning_rate in ((0, 1), (1, 0.5)):
        sample_weights = np.ones(m) / m
        plt.sca(axes[subplot])
        for i in range(5):
            svm_clf = SVC(kernel="rbf", C=0.2, gamma=0.6, random_state=42)
            svm_clf.fit(X_train, y_train, sample_weight=sample_weights * m)
            y_pred = svm_clf.predict(X_train)

            r = sample_weights[y_pred != y_train].sum() / sample_weights.sum()  # calculates weighted error rate
            alpha = learning_rate * np.log((1-r) / r)  # calculate predictor weight
            sample_weights[y_pred != y_train] *= np.exp(alpha)  # update weights to boost weights of misclassified items
            sample_weights /= sample_weights.sum()  # normalization step

            du.plot_decision_boundary(svm_clf, X, y, alpha=0.2)
            plt.title("learning_rate={}".format(learning_rate), fontsize=16)

        if subplot == 0:
            plt.text(-0.75, -0.95, "1", fontsize=14)
            plt.text(-1.05, -0.95, "2", fontsize=14)
            plt.text(1.0, -0.95, "3", fontsize=14)
            plt.text(-1.45, -0.5, "4", fontsize=14)
            plt.text(1.36, -0.95, "5", fontsize=14)
        else:
            plt.ylabel("")

    plt.show();



def run():
    X_train, X_test, y_train, y_test = du.get_moons()
    X, y = du.get_raw_moons()
    # plot_adaboost(X=X, y=y, X_train=X_train, y_train=y_train)
    plot_consecutive(X=X, y=y, X_train=X_train, y_train=y_train)
