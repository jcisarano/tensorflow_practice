import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def generate_data(random_seed=42):
    np.random.seed(random_seed)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    return X, y


def plot_data(X_train, y_train, X_test=None, y_pred=None, axis=[0, 2, 0, 15]):
    plt.plot(X_train, y_train, "b.")
    if X_test is not None and y_pred is not None:
        plt.plot(X_test, y_pred, "r-", label="Predictions")
        plt.legend(loc="upper left", fontsize=14)
    plt.xlabel("$X_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis(axis)
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
    X_b = np.c_[np.ones(X.shape), X]
    for iteration in range(n_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
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
        gradients = 2/m * X_train_b.T.dot(X_train_b.dot(theta) - y_train)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)


def learning_schedule(t, t0, t1):
    return t0 / (t + t1)


def stochastic_gradient_descent(X, y, X_test, theta, m, theta_path_sgd, n_epochs=50, t0=5, t1=50):
    X_b = np.c_[np.ones(X.shape), X]
    X_test_b = np.c_[np.ones(X_test.shape), X_test]
    for epoch in range(n_epochs):
        for i in range(m):
            if epoch == 0 and i < 20:
                y_predict = X_test_b.dot(theta)
                style = "b-" if i > 0 else "r--"
                plt.plot(X_test, y_predict, style)
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2*xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(epoch*m+i, t0, t1)
            theta = theta - eta*gradients
            theta_path_sgd.append(theta)
    plt.plot(X, y, "b.")
    plt.xlabel("$X_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.show()


def do_sgd(X, y):
    from sklearn.linear_model import SGDRegressor
    sgd_reg = SGDRegressor(max_iter=50, tol=-np.infty, penalty=None, eta0=0.1, random_state=42)
    sgd_reg.fit(X, y.ravel())
    print(sgd_reg.intercept_, sgd_reg.coef_)


def mini_batch_gradient_descent(X, y, theta, theta_path_mgd, m, n_iterations=50, minibatch_size=20, t0=200, t1=1000):
    X_b = np.c_[np.ones(X.shape), X]
    t = 0
    for epoch in range(n_iterations):
        shuffled_indices = np.random.permutation(m)
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0, m, minibatch_size):
            t += 1
            xi = X_b_shuffled[i:i+minibatch_size]
            yi = y_shuffled[i:i+minibatch_size]
            gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(t, t0, t1)
            theta = theta - eta*gradients
            theta_path_mgd.append(theta)

    # print(theta)

def plot_all_gradient_descent(sgd_path, mgd_path, bgd_path):
    plt.figure(figsize=(10, 7))
    plt.plot(sgd_path[:, 0], sgd_path[:, 1], "r-s", linewidth=1, label="Stochastic")
    plt.plot(mgd_path[:, 0], mgd_path[:, 1], "g-+", linewidth=1, label="Mini-batch")
    plt.plot(bgd_path[:, 0], bgd_path[:, 1], "b-o", linewidth=1, label="Batch")
    plt.legend(loc="upper left", fontsize=16)
    plt.xlabel(r"$\theta_0$", fontsize=20)
    plt.ylabel(r"$\theta_1$", fontsize=20, rotation=0)
    plt.axis([2.5, 4.5, 2.3, 3.9])
    plt.show()


def do_polynomial_regression(X, y, X_test):
    from sklearn.preprocessing import PolynomialFeatures
    m = len(X)
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    # print(X[0])
    # print(X_poly[0])
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    # print(lin_reg.intercept_, lin_reg.coef_)
    X_test_poly = poly_features.transform(X_test)
    y_pred = lin_reg.predict(X_test_poly)
    return y_pred


def do_polynomial_regression_compare(X_train, y_train, X_test):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    for style, width, degree in (("g-", 1, 300), ("b--", 2, 2), ("r-+", 2, 1)):
        polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
        std_scaler = StandardScaler()
        lin_reg = LinearRegression()
        polynomial_regression = Pipeline([
            ("poly_features", polybig_features),
            ("std_scaler", std_scaler),
            ("lin_reg", lin_reg),
        ])
        polynomial_regression.fit(X_train, y_train)
        y_newbig = polynomial_regression.predict(X_test)
        plt.plot(X_test, y_newbig, style, label=str(degree), linewidth=width)

    plt.plot(X_train, y_train, "b.", linewidth=3)
    plt.legend(loc="upper left")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([-3, 3, 0, 10])
    plt.show()


def run():
    X, y = generate_data()
    # plot_data(X, y)
    t_b = calc_theta_best(X, y)
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
    # listsq() uses pinv() internally:
    print(np.linalg.pinv(X_with_bias_column).dot(y))

    # linear regression using batch gradient descent
    # theta = do_linear_regression_with_batch_gd(X_with_bias_column, y)
    theta = do_linear_regression_with_batch_gd(X, y)
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

    # stochastic gradient descent
    theta_path_sgd = []
    m = len(X_with_bias_column)
    np.random.seed(42)
    theta = np.random.randn(2, 1)

    stochastic_gradient_descent(X, y, X_new, theta, m, theta_path_sgd, n_epochs=50, t0=5, t1=50)
    do_sgd(X, y)

    # mini-batch gradient descent
    np.random.seed(42)
    theta = np.random.randn(2, 1)  # random initialization
    theta_path_mgd = []
    m = len(X)
    mini_batch_gradient_descent(X, y, theta, theta_path_mgd, m, n_iterations=50, minibatch_size=20, t0=200, t1=1000)

    # compare plots of all gradient decent methods so far
    bgd_path = np.array(theta_path_bgd)
    sgd_path = np.array(theta_path_sgd)
    mgd_path = np.array(theta_path_mgd)
    plot_all_gradient_descent(sgd_path, mgd_path, bgd_path)

    # polynomial regression
    # modify a polynomial to use linear regression
    np.random.seed(42)
    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
    X_test = np.linspace(-3, 3, 100).reshape(100, 1)

    y_pred = do_polynomial_regression(X, y, X_test=X_test)
    plot_data(X, y, y_pred=y_pred, X_test=X_test, axis=[-3, 3, 0, 10])

    do_polynomial_regression_compare(X, y, X_test)

