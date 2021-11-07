"""
    Gradient boost trains a series of predictors, each one improving on its predecessor.
    Unlike AdaBoost, it tries to fit the new predictor to the residual error of the previous one.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor


def create_quadratic_dataset():
    np.random.seed(42)
    X = np.random.rand(100, 1) - 0.5
    y = 3*X[:, 0]**2 + 0.05*np.random.randn(100)
    return X, y


def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1,1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=2)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)


def simple_gradient_boost_eg(X, y):
    tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg1.fit(X, y)
    y2 = y - tree_reg1.predict(X)

    tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg2.fit(X, y2)
    y3 = y2 - tree_reg2.predict(X)

    tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg3.fit(X, y3)

    X_new = np.array([[0.8]])
    y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
    print(y_pred)


def run():
    X, y = create_quadratic_dataset()
    simple_gradient_boost_eg(X, y)
