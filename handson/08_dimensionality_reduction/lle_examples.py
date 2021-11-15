"""
Local Linear Embedding is another nonlinear dimensionality reduction technique
It does not rely on projections, instead it compares training sets to their closest neighbors
and looks for low-dimension representations that preserve these relationships as best possible.
"""
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.pyplot as plt


def get_data(n_samples=1000, noise=0.2):
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=41)
    return X, t


def plot_lle(X, t):
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
    X_reduced = lle.fit_transform(X)

    plt.title("Unrolled swiss roll using LLE", fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z-2$", fontsize=18)
    plt.axis([-0.065, 0.055, -0.1, 0.12])
    plt.grid(True)
    plt.show()


def run():
    X, t = get_data()
    plot_lle(X, t)

