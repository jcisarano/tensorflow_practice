from timeit import timeit

from sklearn.datasets import load_iris, fetch_openml
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import stats
import numpy as np

from sklearn.cluster import MiniBatchKMeans

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from visualization_helpers import plot_clusters, plot_data, plot_centroids, plot_decision_boundaries, \
    plot_clusterer_comparison


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


def compare_clusterers():
    X, _ = create_blobs()
    kmeans_rnd_1 = KMeans(n_clusters=5, init="random", n_init=1,
                          algorithm="full", random_state=2)
    kmeans_rnd_2 = KMeans(n_clusters=5, init="random", n_init=1,
                          algorithm="full", random_state=5)

    plot_clusterer_comparison(kmeans_rnd_1, kmeans_rnd_2, X, "Solution one", "Solution two (with different random seed")
    plt.show()


def kmeans_init_example():
    """
    By setting n_init higher, the system will run multiple times and use the best results, the one with the lowest
    inertia value. Inertia is is the sum of the squared distances between each training instance and its closest
    centroid.
    :return:
    """
    X, _ = create_blobs()
    kmeans_rnd_10_init = KMeans(n_clusters=5, init="random", n_init=10,
                                algorithm="full", random_state=2)
    kmeans_rnd_10_init.fit(X)

    plt.figure(figsize=(8, 4))
    plot_decision_boundaries(kmeans_rnd_10_init, X)
    plt.show()

    # inertia info
    print(kmeans_rnd_10_init.inertia_)
    # same as squared distance between training instance and closest centroid:
    X_dist = kmeans_rnd_10_init.transform(X)
    print(np.sum(X_dist[np.arange(len(X_dist)), kmeans_rnd_10_init.labels_] ** 2))


def kmeans_plusplus_example():
    """
    K-Means++ uses a better means of choosing the starting centroids (the rest of the algorithm is unchanged).
    The first centroid is chosen at random, the rest are farther away from already chosen centroids.
    This is the default init method, but it can be specified as init="k-means++"
    :return:
    """

    X, _ = create_blobs()
    # or init with your own points:
    good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
    kmeans = KMeans(n_clusters=5, init=good_init, n_init=1, random_state=42)
    kmeans.fit(X)
    print(kmeans.inertia_)
    plt.figure(figsize=(8, 4))
    plot_decision_boundaries(kmeans, X)
    plt.show()


def load_next_batch(X, batch_size):
    return X[np.random.choice(len(X), batch_size, replace=False)]


def kmeans_mini_batch():
    X, _ = create_blobs()

    minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
    minibatch_kmeans.fit(X)
    print(minibatch_kmeans.inertia_)

    # memmap minibatch
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    mnist.target = mnist.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(mnist["data"], mnist["target"], random_state=42)
    filename = "data/my_mnist.data"
    X_mm = np.memmap(filename, dtype="float32", mode="write", shape=X_train.shape)
    X_mm[:] = X_train

    minibatch_kmeans = MiniBatchKMeans(n_clusters=10, batch_size=10, random_state=42)
    minibatch_kmeans.fit(X_mm)


def kmeans_minibatch_manual():
    X, _ = create_blobs()
    np.random.seed(42)

    k = 5
    n_init = 10
    n_iterations = 100
    batch_size = 100
    init_size = 500
    evaluate_on_last_n_iters = 10
    best_kmeans = None

    for init in range(n_init):
        minibatch_kmeans = MiniBatchKMeans(n_clusters=k, init_size=init_size)
        X_init = load_next_batch(X, batch_size=init_size)
        minibatch_kmeans.partial_fit(X_init)

        minibatch_kmeans.sum_inertia_ = 0
        for iteration in range(n_iterations):
            X_batch = load_next_batch(X, batch_size)
            minibatch_kmeans.partial_fit(X_batch)
            if iteration >= n_iterations - evaluate_on_last_n_iters:
                minibatch_kmeans.sum_inertia_ += minibatch_kmeans.inertia_

        if (best_kmeans is None or
            minibatch_kmeans.sum_inertia_ < best_kmeans.inertia_):
            best_kmeans = minibatch_kmeans

    print(best_kmeans.score(X))


def plot_minibatch_train_times():
    X, _ = create_blobs()

    times = np.empty((100, 2))
    inertias = np.empty((100, 2))
    for k in range(1, 101):
        kmeans_ = KMeans(n_clusters=k, random_state=42)
        minibatch_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
        print("\r{}/{}".format(k, 100), end="")
        times[k-1, 0] = timeit("kmeans_.fit(X)", number=10, globals=locals())
        times[k-1, 1] = timeit("minibatch_kmeans.fit(X)", number=10, globals=locals())
        inertias[k-1, 0] = kmeans_.inertia_
        inertias[k-1, 1] = minibatch_kmeans.inertia_

    plt.figure(figsize=(10, 4))

    plt.subplot(121)
    plt.plot(range(1, 101), inertias[:, 0], "r--", label="K-means")
    plt.plot(range(1, 101), inertias[:, 1], "b.-", label="Mini batch K-means")
    plt.xlabel("$k$", fontsize=16)
    plt.title("Inertia", fontsize=14)
    plt.legend(fontsize=14)
    plt.axis([1, 100, 0, 100])

    plt.subplot(122)
    plt.plot(range(1, 101), times[:, 0], "r--", label="K-means")
    plt.plot(range(1, 101), times[:, 1], "b.-", label="Mini batch K-means")
    plt.xlabel("$k$", fontsize=16)
    plt.title("Training time (seconds)", fontsize=14)
    plt.axis([1, 100, 0, 6])
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = load_iris()
    # show_iris_clusters(data)
    # predict_iris(data)
    # plot_blobs()
    # predict_blobs()
    # compare_kmeans_diff_iter()
    # compare_clusterers()
    # kmeans_init_example()
    # kmeans_plusplus_example()
    # kmeans_mini_batch()
    # kmeans_minibatch_manual()
    plot_minibatch_train_times()




