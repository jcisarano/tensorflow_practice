import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering

from dbscan_exercises import create_moons


def simple_example(X):
    sc1 = SpectralClustering(n_clusters=2, gamma=100, random_state=42)
    sc1.fit(X)

    sc2 = SpectralClustering(n_clusters=2, gamma=1, random_state=42)
    sc2.fit(X)

    print(np.percentile(sc1.affinity_matrix_, 95))

    plt.figure(figsize=(9, 3.2))
    plt.subplot(121)
    plot_spectral_clustering(sc1, X, size=500, alpha=0.1)

    plt.subplot(122)
    plot_spectral_clustering(sc2, X, size=4000, alpha=0.01, show_ylabels=False)

    plt.show()


def plot_spectral_clustering(sc, X, size, alpha, show_xlabels=True, show_ylabels=True):
    plt.scatter(X[:, 0], X[:, 1], marker="o", s=size, c="gray", cmap="Paired", alpha=alpha)
    plt.scatter(X[:, 0], X[:, 1], marker="o", s=30, c="w")
    plt.scatter(X[:, 0], X[:, 1], marker=".", c=sc.labels_, cmap="Paired")

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)

    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14)
    else:
        plt.tick_params(labelleft=False)
    plt.title("RBF gamma={}".format(sc.gamma), fontsize=14)

def run():
    X, y = create_moons()
    simple_example(X)


