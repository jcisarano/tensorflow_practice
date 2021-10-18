import matplotlib.pyplot as plt
import numpy as np
# from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.pipeline import Pipeline

import utils


def create_linear_svr(X, y, epsilon=1.5):
    svm_reg = LinearSVR(epsilon=epsilon, random_state=42)
    svm_reg.fit(X, y)
    return svm_reg


def find_support_vectors(svm_reg, X, y):
    y_pred = svm_reg.predict(X)
    off_margin = (np.abs(y - y_pred) >= svm_reg.epsilon)
    return np.argwhere(off_margin)


def plot_svm_regression(svm_reg, X, y, axes):
    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
    y_pred = svm_reg.predict(x1s)
    plt.plot(x1s, y_pred, "k-", linewidth=2, label=r"$\hat{y}$")
    plt.plot(x1s, y_pred + svm_reg.epsilon, "k--")
    plt.plot(x1s, y_pred - svm_reg.epsilon, "k--")
    plt.scatter(X[svm_reg.support_], y[svm_reg.support_], s=180, facecolors="#FFAAAA")
    plt.plot(X, y, "bo")
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.legend(loc="upper left", fontsize=18)
    plt.axis(axes)


def do_svm_regression(X, y):
    """
    svm regression using kernel trick on linear data
    :param X:
    :param y:
    :return:
    """
    svm_reg1 = create_linear_svr(X=X, y=y)
    svm_reg2 = create_linear_svr(X=X, y=y, epsilon=0.5)
    svm_reg1.support_ = find_support_vectors(svm_reg1, X, y)
    svm_reg2.support_ = find_support_vectors(svm_reg2, X, y)
    eps_x1 = 1
    eps_y_pred = svm_reg1.predict([[eps_x1]])
    # print(eps_y_pred)

    fig, axes = plt.subplots(ncols=2, figsize=(9, 4), sharey=True)
    plt.sca(axes[0])
    plot_svm_regression(svm_reg1, X, y, [0, 2, 3, 11])
    plt.title(r"$\epsilon = {}$".format(svm_reg1.epsilon), fontsize=18)
    plt.ylabel(r"$y$", fontsize=18, rotation=0)
    plt.annotate("", xy=(eps_x1, eps_y_pred), xycoords="data",
                 xytext=(eps_x1, eps_y_pred - svm_reg1.epsilon),
                 textcoords="data", arrowprops={"arrowstyle": "<->", "linewidth": 1.5}
                 )
    plt.text(0.91, 5.6, r"$\epsilon$", fontsize=20)
    plt.sca(axes[1])
    plot_svm_regression(svm_reg2, X, y, [0, 2, 3, 11])
    plt.title(r"$\epsilon = {}$".format(svm_reg2.epsilon), fontsize=18)
    plt.show()



def run():
    np.random.seed(42)
    m = 50
    X = 2 * np.random.rand(m, 1)  # shape (50,1)
    y = (4 + 3 * X + np.random.randn(m, 1)).ravel()  # shape(50,)
    # print(X)
    # print(y)

    # model_svmr = create_linear_svr(X, y)
    # print(model_svmr)

    do_svm_regression(X, y)
