from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def get_mnist(m=10000):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, cache=True)
    mnist.target = mnist.target.astype(np.uint8)

    idx = np.random.permutation(60000)[:m]

    X = mnist["data"][idx]
    y = mnist["target"][idx]

    return X, y


def scatter_plot(X_reduced, y):
    plt.figure(figsize=(10, 7))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="jet")
    plt.colorbar()
    plt.axis("off")
    plt.show()


def plot_subset(X_reduced, y):
    plt.figure(figsize=(9, 9))
    cmap = plt.cm.get_cmap("jet")
    for digit in (2, 3, 5):
        plt.scatter(X_reduced[y == digit, 0], X_reduced[y == digit, 1], c=[cmap(digit / 9)])
    plt.axis("off")
    plt.show()


def improve_2_3_5(X_reduced, X, y):
    # vals 2, 3, and 5 are scattered all over the place
    # use this function to see them separate from the others
    plot_subset(X_reduced, y)

    # now try to improve their distribution by reducing them separately from the others
    idx = (y == 2) | (y == 3) | (y == 5)
    X_subset = X[idx]
    y_subset = y[idx]
    tsne_subset = TSNE(n_components=2, learning_rate="auto", random_state=42)
    X_subset_reduced = tsne_subset.fit_transform(X_subset)
    plot_subset(X_subset_reduced, y_subset)


def plot_digits(X, y, min_distance=0.05, images=None, figsize=(13, 10)):
    X_normalized = MinMaxScaler().fit_transform(X)
    neighbors = np.array([[10., 10.]])
    plt.figure(figsize=figsize)
    cmap = plt.cm.get_cmap("jet")
    digits = np.unique(y)
    for digit in digits:
        plt.scatter(X_normalized[y == digit, 0], X_normalized[y == digit, 1], c=[cmap(digit / 9)])
    plt.axis("off")

    plt.show()


def run():
    X, y = get_mnist()
    # classes = np.unique(y)

    tsne = TSNE(n_components=2, learning_rate="auto", random_state=42)
    X_reduced = tsne.fit_transform(X)

    # norm = plt.Normalize(vmin=0, vmax=9)
    # c = norm(y)
    # scatter_plot(X_reduced, c)

    # improve_2_3_5(X_reduced, X, y)

    plot_digits(X_reduced, y)

