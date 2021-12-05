import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN


def create_moons(n_samples=1000, noise=0.05, random_state=42):
    return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)


def simple_example(X, y):
    dbscan = DBSCAN(eps=0.05, min_samples=5)
    dbscan.fit(X)

    print(dbscan.labels_[:10])
    print(len(dbscan.core_sample_indices_))
    print(dbscan.core_sample_indices_[:10])
    print(dbscan.components_[:3])
    print(np.unique(dbscan.labels_))

    dbscan2 = DBSCAN(eps=0.2)
    dbscan2.fit(X)

    plt.figure(figsize=(9, 3.2))

    plt.subplot(121)
    plot_dbscan(dbscan, X, size=100)

    plt.subplot(122)
    plot_dbscan(dbscan2, X, size=100, show_ylabels=False)

    plt.show()


def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    non_core_mask = ~(core_mask | anomalies_mask)

    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]

    # first draw background colors
    # plt.scatter(cores[:, 0], cores[:, 1], marker="o", s=size, c=dbscan.labels_[core_mask], cmap="Paired")

    # next plot core items on top of background samples with smaller size and contrasting color
    # plt.scatter(cores[:, 0], cores[:, 1], marker="*", s=20, c=dbscan.labels_[core_mask])

    # plot the anomalies in big red Xs
    # plt.scatter(anomalies[:, 0], anomalies[:, 1], marker="x", s=100, c="r")

    # plot the non core items
    plt.scatter(non_cores[:, 0], non_cores[:, 1], marker=".", c=dbscan.labels_[non_core_mask])
    
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)

    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14)
    else:
        plt.tick_params(labelleft=False)
    plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)





def run():
    X, y = create_moons()
    simple_example(X, y)
