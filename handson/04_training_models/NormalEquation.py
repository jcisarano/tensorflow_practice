import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


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
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
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
        gradients = 2 / m * X_train_b.T.dot(X_train_b.dot(theta) - y_train)
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
            xi = X_b[random_index:random_index + 1]
            yi = y[random_index:random_index + 1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(epoch * m + i, t0, t1)
            theta = theta - eta * gradients
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
            xi = X_b_shuffled[i:i + minibatch_size]
            yi = y_shuffled[i:i + minibatch_size]
            gradients = 2 / minibatch_size * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(t, t0, t1)
            theta = theta - eta * gradients
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


def plot_learning_curves(model, X, y, axis=[0, 80, 0, 3]):
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.axis(axis)
    plt.show()


def graph_linear_vs_polynomial_regression(X, y, X_test, model_class, polynomial, alphas, **model_kargs):
    from sklearn.linear_model import Ridge
    for alpha, style in zip(alphas, ("b-", "g--", "r:")):
        model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
        if polynomial:
            model = Pipeline([
                ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                ("std_scaler", StandardScaler()),
                ("regul_reg", model)
            ])
        model.fit(X, y)
        y_pred_regul = model.predict(X_test)
        lw = 2 if alpha > 0 else 1
        plt.plot(X_test, y_pred_regul, style, linewidth=lw, label=r"$\alpha = {}$".format(alpha))
    plt.plot(X, y, "b.", linewidth=3)
    plt.legend(loc="upper left", fontsize=15)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 3, 0, 4])


def do_manual_early_stop(X_train, X_val, y_train, y_val):
    from sklearn.metrics import mean_squared_error
    poly_scaler = Pipeline([
        ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
        ("std_scaler", StandardScaler()),
    ])
    X_train_poly_scaled = poly_scaler.fit_transform(X_train)
    X_val_poly_scaled = poly_scaler.transform(X_val)

    sgd_reg = SGDRegressor(max_iter=1,
                           tol=-np.infty,
                           penalty=None,
                           eta0=0.0005,
                           warm_start=True,
                           learning_rate="constant",
                           random_state=42)
    n_epochs = 500
    train_errors, val_errors = [], []
    for epoc in range(n_epochs):
        sgd_reg.fit(X_train_poly_scaled, y_train)
        y_train_predict = sgd_reg.predict(X_train_poly_scaled)
        y_val_predict = sgd_reg.predict(X_val_poly_scaled)
        train_errors.append(mean_squared_error(y_train, y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    best_epoch = np.argmin(val_errors)
    best_val_rmse = np.sqrt(val_errors[best_epoch])

    plt.annotate("Best model:",
                 xy=(best_epoch, best_val_rmse),
                 xytext=(best_epoch, best_val_rmse + 1),
                 ha="center",
                 arrowprops=dict(facecolor="black", shrink=0.05),
                 fontsize=16,
                 )

    best_val_rmse -= 0.03  # improves the look of the graph
    plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], "k:", linewidth=2)
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")
    plt.plot(np.sqrt(train_errors), "r--", linewidth=2, label="Training set")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.show()

    # clone best model
    from sklearn.base import clone
    sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True, penalty=None, learning_rate="constant",
                           eta0=0.0005, random_state=42)
    minimum_val_error = float("inf")
    best_epoch = None
    best_model = None
    for epoch in range(1000):
        sgd_reg.fit(X_train_poly_scaled, y_train)
        y_val_predict = sgd_reg.predict(X_val_poly_scaled)
        val_error = mean_squared_error(y_val, y_val_predict)
        if val_error < minimum_val_error:
            minimum_val_error = val_error
            best_epoch = epoch
            best_model = clone(sgd_reg)
    print(best_epoch, best_model)


def bgd_path(theta, X, y, l1, l2, core=1, eta=0.1, n_iterations=50):
    path = [theta]
    for iteration in range(n_iterations):
        gradients = core * 2 / len(X) * X.T.dot(X.dot(theta) - y) + l1 * np.sign(theta) + 2 * l2 * theta
        theta = theta - eta * gradients
        path.append(theta)
    return np.array(path)


def regularization_plots():
    t1a, t1b, t2a, t2b = -1, 3, -1.5, 1.5
    t1s = np.linspace(t1a, t1b, 500)  # returns 500 evenly spaced numbers in range [t1a,t1b]
    t2s = np.linspace(t2a, t2b, 500)  # returns 500 evenly spaced numbers in range [t1a,t1b]
    t1, t2 = np.meshgrid(t1s, t2s)  # creates rectangular grid of array of x-y values
    # print(t1.shape, t2.shape)  # result is (500, 500) (500, 500)
    # print(t1 == t2)  # result is false

    # np.c_ Translates slice objects to concatenation along the second axis.
    # ravel() - returns a contiguous, flattened 1-D array
    T = np.c_[t1.ravel(), t2.ravel()]
    Xr = np.array([[-1, 1], [-0.3, -1], [1, 0.1]])
    yr = 2 * Xr[:, :1] + 0.5 * Xr[:, 1:]

    J = (1 / len(Xr) * np.sum((T.dot(Xr.T) - yr.T) ** 2, axis=1)).reshape(t1.shape)

    N1 = np.linalg.norm(T, ord=1, axis=1).reshape(t1.shape)
    N2 = np.linalg.norm(T, ord=2, axis=1).reshape(t1.shape)

    t_min_idx = np.unravel_index(np.argmin(J), J.shape)
    t1_min, t2_min = t1[t_min_idx], t2[t_min_idx]

    t_init = np.array([[0.25], [-1]])

    plt.figure(figsize=(12, 8))
    for i, N, l1, l2, title in ((0, N1, 0.5, 0, "Lasso"), (1, N2, 0, 0.1, "Ridge")):
        JR = J + l1 * N1 + l2 * N2 ** 2

        tr_min_idx = np.unravel_index(np.argmin(JR), JR.shape)
        t1r_min, t2r_min = t1[tr_min_idx], t2[tr_min_idx]

        levelsJ = (np.exp(np.linspace(0, 1, 20)) - 1) * (np.max(J) - np.min(J)) + np.min(J)
        levelsJR = (np.exp(np.linspace(0, 1, 20)) - 1) * (np.max(JR) - np.min(JR)) + np.min(JR)
        levelsN = np.linspace(0, np.max(N), 10)

        path_J = bgd_path(t_init, Xr, yr, l1=0, l2=0)
        path_JR = bgd_path(t_init, Xr, yr, l1=l1, l2=l2)
        path_N = bgd_path(t_init, Xr, yr, l1=np.sign(l1) / 3, l2=np.sign(l2), core=0)

        plt.subplot(221 + i * 2)
        plt.grid(True)
        plt.axhline(y=0, color="k")
        plt.axvline(x=0, color="k")
        plt.contourf(t1, t2, J, levels=levelsJ, alpha=0.9)
        plt.contour(t1, t2, N, levels=levelsN)
        plt.plot(path_J[:, 0], path_J[:, 1], "w-o")
        plt.plot(path_N[:, 0], path_N[:, 1], "y-^")
        plt.plot(t1_min, t2_min, "rs")
        plt.title(r"$\ell_{}$ penalty".format(i + 1), fontsize=16)
        plt.axis([t1a, t1b, t2a, t2b])
        if i == 1:
            plt.xlabel(r"$\theta_1$", fontsize=20)
        plt.ylabel(r"$\theta_2$", fontsize=20, rotation=0)
        plt.subplot(222 + i * 2)
        plt.grid(True)
        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')
        plt.contourf(t1, t2, JR, levels=levelsJR, alpha=0.9)
        plt.plot(path_JR[:, 0], path_JR[:, 1], "w-o")
        plt.plot(t1r_min, t2r_min, "rs")
        plt.title(title, fontsize=16)
        plt.axis([t1a, t1b, t2a, t2b])
        if i == 1:
            plt.xlabel(r"$\theta_1$", fontsize=20)
    plt.show()


def plot_log_regression():
    t = np.linspace(-10, 10, 100)
    sig = 1 / (1 + np.exp(-t))
    plt.figure(figsize=(9, 3))
    plt.plot([-10, 10], [0, 0], "k-")
    plt.plot([-10, 10], [0.5, 0.5], "k:")
    plt.plot([-10, 10], [1, 1], "k:")
    plt.plot([-0, 0], [-1.1, 1.1], "k:")
    plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
    plt.xlabel("t")
    plt.legend(loc="upper left", fontsize=20)
    plt.axis([-10, 10, -0.1, 1.1])
    plt.show()


def iris_virginica_fancy_plot(model, X, y):
    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_prob = model.predict_proba(X_new)  # predict_proba() gives the probability of each output, 0 and 1, so (p0, p1)
    decision_boundary = X_new[y_prob[:, 1] >= 0.5][0]

    plt.figure(figsize=(8, 5))
    plt.plot(X[y == 0], y[y == 0], "bs")
    plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
    plt.plot(X_new, y_prob[:, 1], "g-", linewidth=2, label="Iris-Virginica")
    plt.plot(X_new, y_prob[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
    plt.text(decision_boundary + 0.02, 0.15, "Decision boundary", fontsize=14, color="k", ha="center")
    plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc="b", ec="b")
    plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc="g", ec="g")
    plt.xlabel("Petal width (cm)", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    plt.legend(loc="center left", fontsize=14)
    plt.axis([0, 3, -0.02, 1.02])

    plt.show()


def logistic_regression_plot(iris):
    from sklearn.linear_model import LogisticRegression
    X = iris["data"][:, (2, 3)]  # now use petal length and width
    y = (iris["target"] == 2).astype(np.int)
    log_reg = LogisticRegression(solver="liblinear", C=10 ** 10, random_state=42)
    log_reg.fit(X, y)

    x0, x1 = np.meshgrid(
        np.linspace(2.9, 7, 500).reshape(-1, 1),
        np.linspace(0.8, 7, 200).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]  # c_() translates slice objects to concatenation along the second axis.
    y_proba = log_reg.predict_proba(X_new)

    plt.figure(figsize=(10, 5))
    plt.plot(X[y == 0, 0], X[y == 0, 1], "bs")
    plt.plot(X[y == 1, 0], X[y == 1, 1], "g^")

    zz = y_proba[:, 1].reshape(x0.shape)
    contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)

    left_right = np.array([2.9, 7])
    boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]

    plt.clabel(contour, inline=1, fontsize=12)
    plt.plot(left_right, boundary, "k--", linewidth=3)
    plt.text(3.5, 1.5, "Not Iris-Virginica", fontsize=14, color="b", ha="center")
    plt.text(6.5, 2.3, "Iris-Virginica", fontsize=14, color="g", ha="center")
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.axis([2.9, 7, 0.8, 2.7])
    plt.show()


def run():
    X, y = generate_data()
    # plot_data(X, y)
    t_b = calc_theta_best(X, y)
    # print(t_b)

    """
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
    """

    """
    # polynomial regression
    # modify a polynomial to use linear regression
    np.random.seed(42)
    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
    X_test = np.linspace(-3, 3, 100).reshape(100, 1)

    y_pred = do_polynomial_regression(X, y, X_test=X_test)
    plot_data(X, y, y_pred=y_pred, X_test=X_test, axis=[-3, 3, 0, 10])

    # examine performance of different polynomial values on same data set:
    do_polynomial_regression_compare(X, y, X_test)

    # compare learning curves on subset of data to determine performance
    # standard linear regression underfits the polynomial curve:
    # the errors on the train and test sets are close and high
    lin_reg = LinearRegression()
    plot_learning_curves(lin_reg, X, y)

    # learning curve of polynomial regression shows better fit:
    # error rate is much lower than linear regression
    # however, the error rates are farther apart. Training error is lower, so it is overfitting.
    # Using more training data should fix this
    polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])
    plot_learning_curves(polynomial_regression, X, y)
    """

    """Regularized Models"""
    """
    # Ridge regularization with linear regression vs polynomial regression
    from sklearn.linear_model import Ridge
    np.random.seed(42)
    m = 20
    X = 3*np.random.rand(m, 1)
    y = 1 + 0.5*X + np.random.randn(m, 1)/1.5
    X_test = np.linspace(0, 3, 100).reshape(100, 1)

    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    graph_linear_vs_polynomial_regression(X, y, X_test, Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(122)
    graph_linear_vs_polynomial_regression(X, y, X_test, Ridge, polynomial=True, alphas=(0, 10 ** -5, 1), random_state=42)
    plt.show()

    # Ridge Regression built-in closed form solution
    ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
    ridge_reg.fit(X, y)
    print(ridge_reg.predict([[1.5]]))

    # Ridge Regression built-in Stochastic Gradient Descent
    # setting penalty="l2" makes this Ridge Regression
    sgd_reg = SGDRegressor(max_iter=50, tol=-np.infty, penalty="l2", random_state=42)
    sgd_reg.fit(X, y.ravel())
    print(sgd_reg.predict([[1.5]]))

    # solver=sag, Stochastic Average Gradient descent,
    ridge_reg = Ridge(alpha=1, solver="sag", random_state=42)
    ridge_reg.fit(X, y)
    print(ridge_reg.predict([[1.5]]))
    """

    """Lasso Regression"""
    """
    from sklearn.linear_model import Lasso
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    graph_linear_vs_polynomial_regression(X, y, X_test, Lasso, polynomial=False, alphas=(0, 0.1, 1), random_state=42)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(122)
    graph_linear_vs_polynomial_regression(X, y, X_test, Lasso, polynomial=True, alphas=(0, 10**-7, 1), tol=1, random_state=42)
    plt.show()

    lasso_reg = Lasso(alpha=0.1)
    lasso_reg.fit(X, y)
    print(lasso_reg.predict([[1.5]]))
    """

    """Elastic net"""
    """
    from sklearn.linear_model import ElasticNet
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    elastic_net.fit(X, y)
    print(elastic_net.predict([[1.5]]))

    np.random.seed(42)
    m = 100
    X = 6*np.random.rand(m, 1) - 3
    y = 2 + X + 0.5*X**2 + np.random.randn(m, 1)
    X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)
    do_manual_early_stop(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    """

    # regularization_plots()
    # plot_log_regression()

    from sklearn import datasets
    iris = datasets.load_iris()
    # print(list(iris.keys()))  # list() returns a python list
    # print(iris.DESCR)

    X = iris["data"][:, 3:]  # petal widths
    y = (iris["target"] == 2).astype(np.int)  # 1 if type is Iris-Virginica, otherwise 0

    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(solver="liblinear", random_state=42)
    log_reg.fit(X, y)

    # iris_virginica_fancy_plot(log_reg, X, y)
    print(log_reg.predict([[1.7], [1.5]]))

    logistic_regression_plot(iris)
