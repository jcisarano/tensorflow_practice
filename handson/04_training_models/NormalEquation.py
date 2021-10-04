import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def generate_data(random_seed=42):
    np.random.seed(random_seed)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    return X, y


def plot_data(X, y, X_pred=None, y_pred=None):
    plt.plot(X, y, "b.")
    if X_pred is not None and y_pred is not None:
        plt.plot(X_pred, y_pred, "r-", label="Predictions")
        plt.legend(loc="upper left", fontsize=14)
    plt.xlabel("$X_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.show()


# use Normal Equation to calculate theta best
def calc_theta_best(X, y):
    X_b = np.c_[np.ones((100, 1)), X]  # adds a first column filled with 1, needed for bias
    return np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)


def do_linear_regression(X_train, y_train, X_test):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    # print(lin_reg.intercept_, lin_reg.coef_)
    return lin_reg.predict(X_test)


def do_linear_regression_with_batch_gd(X, y, eta=0.1, n_iterations=1000, m=100):
    theta = np.random.randn(2, 1)
    for iteration in range(n_iterations):
        gradients = 2/m * X.T.dot(X.dot(theta) - y)
        theta = theta - eta * gradients
    return theta


def plot_gradient_descent(X_train, y_train, X_train_b, X_test, X_test_b, theta, eta, theta_path=None):
    m = len(X_train_b)
    plt.plot(X_train, y_train, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_test_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_test, y_predict, style)
        gradients = 2/m * X_test_b.T.dot(X_test_b.dot(theta) - y_train)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)

def run():
    X, y = generate_data()
    # plot_data(X, y)
    t_b = calc_theta_best(X,y)
    # print(t_b)

    # make predictions using theta best
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]
    y_predict = X_new_b.dot(t_b)
    print(y_predict)
    plot_data(X, y, X_new, y_predict)

    print(do_linear_regression(X, y, X_new))

    # LinearRegression() uses listsq() internally:
    X_with_bias_column = np.c_[np.ones((100, 1)), X]
    theta_best_svd, residuals, rank, x = np.linalg.lstsq(X_with_bias_column, y, rcond=1e-6)
    print(theta_best_svd)
    # listsq() uses this internally::
    print(np.linalg.pinv(X_with_bias_column).dot(y))

    # linear regression using batch gradient descent
    theta = do_linear_regression_with_batch_gd(X_with_bias_column, y)
    print(theta)
    print(X_new_b.dot(theta))

    np.random.seed(42)
    theta_path_bgd = []
    theta = np.random.randn(2, 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(131)
    plot_gradient_descent(X, y, X_with_bias_column, X_new, X_new_b, theta, eta=0.02)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(132)
    plot_gradient_descent(X, y, X_with_bias_column, X_new, X_new_b, theta, eta=0.1, theta_path=theta_path_bgd)
    plt.subplot(133)
    plot_gradient_descent(X, y, X_with_bias_column, X_new, X_new_b, theta, eta=0.5)
    plt.show()
