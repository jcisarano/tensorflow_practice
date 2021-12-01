from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import stats
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from visualization_helpers import plot_clusters, plot_data, plot_centroids, plot_decision_boundaries


def show_iris_clusters(data):
    x = data.data
    y = data.target
    # print(data.target_names)

    plt.figure(figsize=(9, 3.5))
    plt.subplot(121)
    plt.plot(x[y == 0, 2], x[y == 0, 3], "yo", label="Iris setosa")
    plt.plot(x[y == 1, 2], x[y == 1, 3], "bs", label="Iris veriscolor")
    plt.plot(x[y == 2, 2], x[y == 2, 3], "g^", label="Iris virginica")
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(fontsize=12)

    plt.subplot(122)
    plt.scatter(x[:, 2], x[:, 3], c="k", marker=".")
    plt.xlabel("Petal length", fontsize=14)
    plt.tick_params(labelleft=False)
    plt.show()


def predict_iris(iris_data):
    X = iris_data.data
    y = iris_data.target

    y_pred = GaussianMixture(n_components=3, random_state=42).fit(X).predict(X)
    mapping = {}
    for class_id in np.unique(y):
        mode, _ = stats.mode(y_pred[y == class_id])
        mapping[mode[0]] = class_id
    print(mapping)
    y_pred = np.array([mapping[cluster_id] for cluster_id in y_pred])

    plt.figure(figsize=(7, 5))
    plt.plot(X[y_pred == 0, 2], X[y_pred == 0, 3], "yo", label="Cluster 1")
    plt.plot(X[y_pred == 1, 2], X[y_pred == 1, 3], "bs", label="Cluster 2")
    plt.plot(X[y_pred == 2, 2], X[y_pred == 2, 3], "g^", label="Cluster 3")
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="upper left", fontsize=12)
    plt.show()

    print("Percent correct predictions", sum(y_pred == y) / len(y))


def create_blobs():
    blob_centers = np.array(
        [[0.2, 2.3],
         [-1.5, 2.3],
         [-2.8, 1.8],
         [-2.8, 2.8],
         [-2.8, 1.3]])
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    return make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=7)


def plot_blobs():
    X, _ = create_blobs()
    plt.figure(figsize=(8, 4))
    plot_clusters(X)
    plt.show()


def predict_blobs():
    X, _ = create_blobs()
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(X)

    # view centroids:
    print(kmeans.cluster_centers_)
    plt.figure(figsize=(8, 4))
    plot_decision_boundaries(kmeans, X)
    plt.show()

    # predict against new points:
    X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
    print(kmeans.predict(X_new))

    # their distance to centroids:
    print(kmeans.transform(X_new))
    # same as
    print(np.linalg.norm(np.tile(X_new, (1, k)).reshape(-1, k, 2) - kmeans.cluster_centers_, axis=2))


def compare_kmeans_diff_iter():
    X, _ = create_blobs()
    kmeans_iter1 = KMeans(n_clusters=5, init="random", n_init=1,
                          algorithm="full", max_iter=1, random_state=0)
    kmeans_iter2 = KMeans(n_clusters=5, init="random", n_init=1,
                          algorithm="full", max_iter=2, random_state=0)
    kmeans_iter3 = KMeans(n_clusters=5, init="random", n_init=1,
                          algorithm="full", max_iter=3, random_state=0)
    kmeans_iter1.fit(X)
    kmeans_iter2.fit(X)
    kmeans_iter3.fit(X)

    plt.figure(figsize=(10, 8))

    plt.subplot(321)
    plot_data(X)
    plot_centroids(kmeans_iter1.cluster_centers_, circle_color="r", cross_color="w")
    plt.ylabel("$x_2$", fontsize=14, rotation=0)
    plt.tick_params(labelbottom=False)
    plt.title("Update the centroids (initially random)", fontsize=14)

    plt.subplot(322)
    plot_data(X)
    plot_decision_boundaries(kmeans_iter1, X, show_xlabels=False, show_ylabels=False)
    plt.tick_params(labelbottom=False)
    plt.title("Label the instances", fontsize=14)

    plt.subplot(323)
    plot_decision_boundaries(kmeans_iter1, X, show_centroids=False, show_xlabels=False)
    plot_centroids(kmeans_iter2.cluster_centers_)

    plt.subplot(324)
    plot_decision_boundaries(kmeans_iter2, X, show_ylabels=False, show_xlabels=False)

    plt.subplot(325)
    plot_decision_boundaries(kmeans_iter2, X, show_centroids=False)
    plot_centroids(kmeans_iter3.cluster_centers_)

    plt.subplot(326)
    plot_decision_boundaries(kmeans_iter3, X, show_ylabels=False)

    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = load_iris()
    # show_iris_clusters(data)
    # predict_iris(data)
    # plot_blobs()
    # predict_blobs()
    compare_kmeans_diff_iter()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
