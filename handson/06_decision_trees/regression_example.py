import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

import visualization as vv


def gen_quad_training_set():
    np.random.seed(42)
    m = 200
    X = np.random.rand(m, 1)
    y = 4 * (X - 0.5)**2
    y = y + np.random.randn(m, 1) / 10

    return X, y


def compare_reg_models(X, y):
    """
    increasing max tree depth from 2 to 3 improves fitting in this case
    :param X:
    :param y:
    :return:
    """
    tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg2 = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree_reg1.fit(X, y)
    tree_reg2.fit(X, y)

    _, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
    plt.sca(axes[0])
    vv.plot_regression_predictions(tree_reg1, X=X, y=y)
    for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
        plt.plot([split, split], [-0.2, 1], style, linewidth=2)
    plt.text(0.21, 0.65, "Depth=0", fontsize=15)
    plt.text(0.01, 0.2, "Depth=1", fontsize=13)
    plt.text(0.65, 0.8, "Depth=1", fontsize=13)
    plt.legend(loc="upper center", fontsize=18)
    plt.title("max_depth=2", fontsize=14)

    plt.sca(axes[1])
    vv.plot_regression_predictions(tree_reg2, X=X, y=y, ylabel=None)
    for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
        plt.plot([split, split], [-0.2, 1], style, linewidth=2)
    for split in (0.0458, 0.1298, 0.2873, 0.9040):
        plt.plot([split, split], [-0.2, 1], "k:", linewidth=1)
    plt.text(0.3, 0.5, "Depth=2", fontsize=13)
    plt.title("max_depth=3", fontsize=14)
    plt.show()

    vv.graphviz_regression_image(tree_reg1)

def run():
    X, y = gen_quad_training_set()
    # tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
    # tree_reg.fit(X, y)

    compare_reg_models(X, y)


