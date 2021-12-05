import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier

from visualization_helpers import plot_decision_boundaries

"""
    DBSCAN defines clusters as continuous regions of high density
    eps is epsilon distance, how far to consider neighboring instances in the same cluster
    min_samples is the minimum number of samples in an area for it to be considered a core instance. It is a dense region.
    All instances in the neighborhood of a core instance belong to the same cluster. More than one core instance can form
        a single cluster.
    Anomalies are any instance not a core instance that does not have a core instance in its neighborhood
"""


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

    # for comparison, another DBSCAN with a larger eps value
    # this one does a much better job of plotting the moons dataset: each moon is a complete cluster
    # and there are only two clusters
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
    anomalies_mask = dbscan.labels_ == -1  # anything labeled as -1 is an anomaly
    non_core_mask = ~(core_mask | anomalies_mask)

    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]

    # first draw background colors
    plt.scatter(cores[:, 0], cores[:, 1], marker="o", s=size, c=dbscan.labels_[core_mask], cmap="Paired")

    # next plot core items on top of background samples with smaller size and contrasting color
    plt.scatter(cores[:, 0], cores[:, 1], marker="*", s=20, c=dbscan.labels_[core_mask])

    # plot the anomalies in big red Xs
    plt.scatter(anomalies[:, 0], anomalies[:, 1], marker="x", s=100, c="r")

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


def knn_prediction(X, y):
    """
    DBSCAN does not have a predict() method, so it us up to the user to decide how to make predictions
    on the trained model. Here we use KNeighborsClassifier
    :return:
    """

    # this matches the second model in the first test, which better fit the moons dataset
    dbscan = DBSCAN(eps=0.2)
    dbscan.fit(X)

    knn = KNeighborsClassifier(n_neighbors=50)
    knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])

    # create a few points to use for predictions
    X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
    print(knn.predict(X_new))
    print(knn.predict_proba(X_new))

    plt.figure(figsize=(6, 3))
    plot_decision_boundaries(knn, X, show_centroids=False)
    plt.scatter(X_new[:, 0], X_new[:, 1], c="b", marker="+", s=200, zorder=10)
    plt.show()

    # knn always makes a prediction, even when the points are far away from the clusters, as with a
    # couple of points in the previous example. It is possible to use kneighbors() function to
    # check against a maximum distance in a prediction:
    y_dist, y_pred_idx = knn.kneighbors(X_new, n_neighbors=1)
    y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]
    y_pred[y_dist > 0.2] = -1
    print(y_pred.ravel())


def run():
    X, y = create_moons()
    # simple_example(X, y)
    knn_prediction(X, y)
