"""
    Gradient boost trains a series of predictors, each one improving on its predecessor.
    Unlike AdaBoost, it tries to fit the new predictor to the residual error of the previous one.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def create_quadratic_dataset():
    np.random.seed(42)
    X = np.random.rand(100, 1) - 0.5
    y = 3*X[:, 0]**2 + 0.05*np.random.randn(100)
    return X, y


def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
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

    # plot
    plt.figure(figsize=(11, 11))

    # plot residuals and tree predictions
    plt.subplot(321)
    plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h_1(x_1)$", style="g-", data_label="Training set")
    plt.ylabel("$y$", fontsize=16, rotation=0)
    plt.title("Residuals and tree predictions", fontsize=16)

    # plot ensemble predictions, using only first tree
    plt.subplot(322)
    plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$", data_label="Training set")
    plt.ylabel("$y$", fontsize=16, rotation=0)
    plt.title("Ensemble predictions", fontsize=16)

    # plot second tree, which was trained on results of first tree
    plt.subplot(323)
    plot_predictions([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", style="g-", data_style="k+", data_label="Residuals")
    plt.ylabel("$y - h_1(x_1)$", fontsize=16)

    # plot ensemble predictions using first and second tree
    plt.subplot(324)
    plot_predictions([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
    plt.ylabel("$y$", fontsize=16, rotation=0)

    # plot third tree, which was trained on results of second tree
    plt.subplot(325)
    plot_predictions([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", style="g-", data_style="k+")
    plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=16)

    # plot ensemble predictions using all three trees
    plt.subplot(326)
    plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
    plt.xlabel("$x_1$", fontsize=16)
    plt.ylabel("$y$", fontsize=16, rotation=0)

    plt.show()


def gradient_boost_regressor(X, y):
    """
    number of predictors (estimators) changes based on learning rate
    need to tune learning rate AND number of predictors
    :param X:
    :param y:
    :return:
    """
    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
    gbrt.fit(X, y)

    gbrt_slow = GradientBoostingRegressor(max_depth=2, n_estimators=200, learning_rate=0.1, random_state=42)
    gbrt_slow.fit(X, y)

    _, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)

    # plot regressor with not enough predictors
    plt.sca(axes[0])
    plot_predictions([gbrt], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="Ensemble predictions")
    plt.title("learning_rate={}, n_estimators={}".format(gbrt.learning_rate, gbrt.n_estimators), fontsize=14)
    plt.xlabel("$x_1$", fontsize=16)
    plt.ylabel("$y$", fontsize=16, rotation=0)

    # plot regressor with too many predictors
    plt.sca(axes[1])
    plot_predictions([gbrt_slow], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
    plt.title("learning_rate={}, n_estimators={}".format(gbrt_slow.learning_rate, gbrt_slow.n_estimators), fontsize=14)
    plt.xlabel("$x_1$", fontsize=16)

    plt.show()


def gb_w_early_stopping(X, y):
    """
    One way to do early stopping with GradientBoostRegressor:
        1. Train GBR with lots of estimators
        2. Check each stage of prediction using staged_predict() to find the lowest error
        3. Train another GBR with only that many estimators
    Note: this requires training a model with lots of estimators, probably many more than you will need.
    :param X:
    :param y:
    :return:
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)
    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
    gbrt.fit(X_train, y_train)

    # checks mse at each stage in prediction to find lowest value, i.e. the best number of predictors
    errors = [mean_squared_error(y_val, y_pred)
              for y_pred in gbrt.staged_predict(X_val)]
    bst_n_estimators = np.argmin(errors) + 1

    # now train another ensemble using only the optimal number of predictors
    gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
    gbrt_best.fit(X_train, y_train)

    min_error = np.min(errors)
    print(min_error)

    plt.figure(figsize=(10, 4))

    # plot the point of the minimum error from gbrt
    plt.subplot(121)
    plt.plot(errors, "b.-")
    plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], "k--")
    plt.plot([0, 120], [min_error, min_error], "k--")
    plt.plot(bst_n_estimators, min_error, "ko")
    plt.text(bst_n_estimators, min_error*1.2, "Minimum", ha="center", fontsize=14)
    plt.axis([0, 120, 0, 0.01])
    plt.xlabel("Number of trees")
    plt.ylabel("Validation error", fontsize=14)

    # now plot the best model (gbrt_best)
    plt.subplot(122)
    plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
    plt.title("Best model (%d trees" % bst_n_estimators, fontsize=14)
    plt.ylabel("$y$", fontsize=16, rotation=0)
    plt.xlabel("$x_1$", fontsize=16)

    plt.show()


def gb_early_stopping_manual(X, y):
    """
    This variation on early stopping checks each estimator as it goes, and it will quit when the mse trends
    worse for five epochs in a row

    Using warm_start in the regressor is important because it keeps existing trees when fit() is called.
    :return:
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)
    gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True, random_state=42)

    min_val_error = float("inf")
    error_going_up = 0
    for n_estimators in range(1, 120):
        gbrt.n_estimators = n_estimators
        gbrt.fit(X_train, y_train)
        y_pred = gbrt.predict(X_val)
        val_error = mean_squared_error(y_val, y_pred)
        if val_error < min_val_error:
            min_val_error = val_error
            error_going_up = 0
        else:
            error_going_up += 1
            if error_going_up == 5:
                break  # this is your early stopping

    print(gbrt.n_estimators)


def run():
    X, y = create_quadratic_dataset()
    # simple_gradient_boost_eg(X, y)
    # gradient_boost_regressor(X, y)
    # gb_w_early_stopping(X, y)
    gb_early_stopping_manual(X, y)
