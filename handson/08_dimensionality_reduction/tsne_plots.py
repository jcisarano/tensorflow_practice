from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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


def run():
    X, y = get_mnist()
    # classes = np.unique(y)

    tsne = TSNE(n_components=2, init="random")
    X_reduced = tsne.fit_transform(X)

    norm = plt.Normalize(vmin=0, vmax=9)
    c = norm(y)
    scatter_plot(X_reduced, c)


