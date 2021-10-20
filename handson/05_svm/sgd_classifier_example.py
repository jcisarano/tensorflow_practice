import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier

import utils


def run():
    C = 2
    iris = load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, width
    y = (iris["target"] == 2).astype(np.float64).reshape(-1, 1)  # Iris virginica

    sgd_clf = SGDClassifier(loss="hinge", alpha=0.017, max_iter=1000, tol=1e-3, random_state=42)
    sgd_clf.fit(X, y.ravel())

    m = len(X)
    t = y*2 - 1  # -1 if y == 0, +1 if y == 1
    X_b = np.c_[np.ones((m, 1)), X]  # add bias input
    X_b_t = X_b*t
    sgd_theta = np.r_[sgd_clf.intercept_[0], sgd_clf.coef_[0]]
    print(sgd_theta)

    support_vectors_idx = (X_b_t.dot(sgd_theta) < 1).ravel()
    sgd_clf.support_vectors_ = X[support_vectors_idx]
    sgd_clf.C = C

    yr = y.ravel()
    plt.figure(figsize=(10, 7))
    plt.plot(X[:, 0][yr == 1], X[:, 1][yr == 1], "g^")
    plt.plot(X[:, 0][yr == 0], X[:, 1][yr == 0], "bs")
    utils.plot_svc_decision_boundary(svm_clf=sgd_clf, xmin=4, xmax=6)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.axis([4, 6, 1, 2.5])
    plt.title("SGDClassifier", fontsize=14)
    plt.show()
