# Exercise:
# Train a LinearSVC on a linearly separable dataset.
# Then train an SVC and a SGDClassifier on the same dataset.
# See if you can get them to produce roughly the same model.
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

import utils


def plot_decision_boundary(clf, xmin, xmax, format="k-"):
    w = clf.coef_[0]
    b = clf.intercept_[0]

    # at the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    plt.plot(x0, decision_boundary, format, linewidth=2)


def plot_decision_boundary_scaled(clf, scaler, format="k-", label="LInearSVC"):
    w = -clf.coef_[0, 0] / clf.coef_[0, 1]
    b = -clf.intercept_[0] / clf.coef_[0, 1]

    line = scaler.inverse_transform([[-10, -10*w + b], [10, 10*w + b]])
    plt.plot(line[:, 0], line[:, 1], format, label=label)


def run():
    # from iris data, use setosa and versicolor because they are linearly separable
    iris = load_iris()
    X = iris["data"][:, (2, 3)]  # only petal length and width
    y = iris["target"]

    setosa_or_versicolor = (y == 0) | (y == 1)
    X = X[setosa_or_versicolor]
    y = y[setosa_or_versicolor]

    C=5
    alpha = 1 / (C*len(X))
    linear_svc_clf = LinearSVC(C=C, loss="hinge", random_state=42)
    svc_clf = SVC(kernel="linear", C=C, verbose=1)
    sgd_clf = SGDClassifier(loss="hinge", learning_rate="constant", alpha=alpha, eta0=1e-3, max_iter=1000,
                            tol=1e-3, random_state=42, verbose=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    linear_svc_clf.fit(X_scaled, y)
    svc_clf.fit(X_scaled, y)
    sgd_clf.fit(X_scaled, y)

    print("LinearSVC intercept: {}, coef:{}".format(linear_svc_clf.intercept_, linear_svc_clf.coef_))
    print("SVC intercept: {}, coef: {}".format(svc_clf.intercept_, svc_clf.coef_))
    print("SGD alpha: {:.5f}, intercept: {}, coef: {}".format(sgd_clf.alpha, sgd_clf.intercept_, sgd_clf.coef_))



    plt.figure(figsize=(11, 4))
    xmin, xmax = -2, 2
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^", label="Iris setosa")
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs", label="Iris versicolor")
    # plot_decision_boundary(clf=linear_svc_clf, xmin=xmin, xmax=xmax, format="k-")
    # plot_decision_boundary(clf=svc_clf, xmin=xmin, xmax=xmax, format="r--")
    # plot_decision_boundary(clf=sgd_clf, xmin=xmin, xmax=xmax, format="b--")

    plot_decision_boundary_scaled(clf=linear_svc_clf, scaler=scaler)
    plot_decision_boundary_scaled(clf=svc_clf, scaler=scaler, format="r--", label="SVC")
    plot_decision_boundary_scaled(clf=sgd_clf, scaler=scaler, format="b:", label="SGDClassifier")

    plt.xlabel("Petal width", fontsize=16)
    plt.ylabel("Petal length", fontsize=16)
    plt.legend(loc="upper left")
    plt.axis([0, 5.5, 0, 2])
    # plt.axis([0, 2, 0, 2])
    plt.show()


