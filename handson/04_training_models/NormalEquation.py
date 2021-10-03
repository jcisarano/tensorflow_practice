import numpy as np
import matplotlib.pyplot as plt


def generate_data(random_seed=42):
    np.random.seed(random_seed)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    return X, y


def plot_data(X, y, X_line=None, y_line=None):
    plt.plot(X, y, "b.")
    if X_line is not None and y_line is not None:
        plt.plot(X_line, y_line, "r-")
    plt.xlabel("$X_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.show()


# use Normal Equation to calculate theta best
def calc_theta_best(X, y):
    X_b = np.c_[np.ones((100, 1)), X]  # adds a first column filled with 1, needed for bias
    return np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)


def run():
    X, y = generate_data()
    plot_data(X, y)
    t_b = calc_theta_best(X,y)
    print(t_b)

    # make predictions using theta best
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]
    y_predict = X_new_b.dot(t_b)
    print(y_predict)
    plot_data(X, y, X_new, y_predict)


