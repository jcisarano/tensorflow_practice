# Examine circles sample dataset from sklearn

from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt


def generate_circles(n_samples=1000):
    X, y = make_circles(n_samples,
                        noise=0.03,
                        random_state=42)
    return X, y


def examine_data(X, y):
    # Check out features and labels:
    print(X[:10])
    print(y[:10])

    circles = pd.DataFrame({"X0:": X[:, 0], "X1": X[:, 1], "label": y})
    print(circles)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.show()

    print(X.shape, y.shape)