import os
from timeit import timeit

from sklearn.datasets import load_iris, fetch_openml, load_digits
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.mixture import GaussianMixture
from scipy import stats
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter
import matplotlib as mpl

from sklearn.cluster import MiniBatchKMeans

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from matplotlib.image import imread
from sklearn.pipeline import Pipeline

from visualization_helpers import plot_clusters, plot_data, plot_centroids, plot_decision_boundaries, \
    plot_clusterer_comparison


PROJECT_ROOT_DIR: str = "."



def kmeans_limits(do_plot=False):
    """
    K-Means does not work well with some data shapes, e.g. the oblong patterns created here.
    :param do_plot:
    :return:
    """
    x1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
    x1 = x1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
    x2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
    x2 = x2 + [6, -8]
    X = np.r_[x1, x2]
    y = np.r_[y1, y2]

    if do_plot:
        plot_clusters(X)
        plt.show()

    return X, y


def kmeans_draw_limits(X, y):
    """
    Neither of these attempts at a K-Means solution fits the data well at all, due to the shape of the clusters.
    :param X:
    :param y:
    :return:
    """
    kmeans_good = KMeans(n_clusters=3, init=np.array([[-1.5, 2.5], [0.5, 0], [4, 0]]), n_init=1, random_state=42)
    kmeans_bad = KMeans(n_clusters=3, random_state=42)
    kmeans_good.fit(X)
    kmeans_bad.fit(X)

    plt.figure(figsize=(10, 3.2))

    plt.subplot(121)
    plot_decision_boundaries(kmeans_good, X)
    plt.title("Inertia = {:.1f}".format(kmeans_good.inertia_), fontsize=14)

    plt.subplot(122)
    plot_decision_boundaries(kmeans_bad, X, show_ylabels=False)
    plt.title("Inertia = {:.1f}".format(kmeans_bad.inertia_), fontsize=14)

    plt.show()


def kmeans_img_segmentation():
    """
    Using K-Means for image segmentation. Setting the number of clusters determines the number of colors that will
    be in the image output, and colors get grouped by similarity and set to an average of their value. Some colors
    will be lost as groups get fewer.
    :return:
    """
    images_path = os.path.join(PROJECT_ROOT_DIR, "images", "unsupervised_learning")
    filename = "ladybug.png"
    image = imread(os.path.join(images_path, filename))
    print("Starting img shape:", image.shape)

    X = image.reshape(-1, 3)
    print("Reshape for clusterer:", X.shape)
    kmeans = KMeans(n_clusters=8, random_state=42).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_img = segmented_img.reshape(image.shape)
    print("Shape after segmentation:", segmented_img.shape)

    segmented_imgs = []
    n_colors = (10, 8, 6, 4, 2)
    for n_clusters in n_colors:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        segmented_img = kmeans.cluster_centers_[kmeans.labels_]
        segmented_img = segmented_img.reshape(image.shape)
        segmented_imgs.append(segmented_img)

    plt.figure(figsize=(10,5))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    plt.subplot(231)
    plt.imshow(image)
    plt.title("Original image")
    plt.axis("off")

    for idx, n_clusters in enumerate(n_colors):
        plt.subplot(232+idx)
        plt.imshow(segmented_imgs[idx])
        plt.title("{} colors".format(n_clusters))
        plt.axis("off")

    plt.show()


def kmeans_for_preprocessing():
    X_digits, y_digits = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)

    # logistic regression model to use as baseline
    log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
    log_reg.fit(X_train, y_train)

    log_reg_score = log_reg.score(X_test, y_test)
    print("Baseline score:", log_reg_score)

    pipeline = Pipeline([
        ("kmeans", KMeans(n_clusters=50, random_state=42)),
        ("log_reg", LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    pipeline_score = pipeline.score(X_test, y_test)
    print("Score with K-Means preprocessing:", pipeline_score)
    print("Percent improvement in score:", 1-(1-pipeline_score) / (1-log_reg_score))


def kmeans_gridsearch():
    """
    Use gridsearch to find best value for K-Mean n_clusters in preprocessing pipeline.
    Note: this function takes a long time to run.
    :return:
    """
    X_digits, y_digits = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)

    pipeline = Pipeline([
        ("kmeans", KMeans(n_clusters=50, random_state=42)),
        ("log_reg", LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)),
    ])

    param_grid = dict(kmeans__n_clusters=range(2, 200))
    grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
    grid_clf.fit(X_train, y_train)

    print("Best params:", grid_clf.best_params_)
    print(grid_clf.score(X_test, y_test))


def kmeans_clustering():
    """
    :return:
    """

    X_digits, y_digits = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)

    # first create baseline LogRegression using only 50 training instances
    n_labeled = 50
    log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=200, random_state=42)
    log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
    print("Baseline score 50 training instances:", log_reg.score(X_test, y_test))

    """
    Now use K-Means to cluster the training set into 50 clusters and find the image for each cluster closest to the
    centroid. These will be used as representative images, the best examples of each cluster.
    """
    k = 50
    kmeans = KMeans(n_clusters=k, random_state=42)
    X_digits_dist = kmeans.fit_transform(X_train)
    representative_digit_idx = np.argmin(X_digits_dist, axis=0)
    X_representative_digits = X_train[representative_digit_idx]

    # visualize the 50 representative images for fun:
    n_cols = 10
    plt.figure(figsize=(8, 2))
    for idx, X_representative_digit in enumerate(X_representative_digits):
        plt.subplot(k // n_cols, n_cols, idx+1)
        plt.imshow(X_representative_digit.reshape(8,8), cmap="binary", interpolation="bilinear")
        plt.axis("off")

    plt.show()

    # create array of labels for the images for training
    y_representative_digits = np.array(y_train[representative_digit_idx])
    print("Representative images labels", y_representative_digits)

    # train using the representative images as training set
    # even though it is still only 50 instances, results improve by almost 9%
    log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=200, random_state=42)
    log_reg.fit(X_representative_digits, y_representative_digits)
    print("Trained with 50 representative images score:", log_reg.score(X_test, y_test))

    # now propagate the label for the representative image to all of the images in the same cluster:
    # it will be a little better than the previous score, about 1%
    # also, it needs a lot more iterations to converge to a result
    y_train_propagated = np.empty(len(X_train))
    for i in range(k):
        y_train_propagated[kmeans.labels_ == i] = y_representative_digits[i]

    log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train_propagated)
    print("Trained with representative image labels propagated to whole cluster:", log_reg.score(X_test, y_test))

    # this time, only propagate the representative labels to the instances closest to the centroid, to avoid
    # mislabeling some outliers
    percentile_closest = 75
    X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
    for i in range(k):
        in_cluster = (kmeans.labels_ == i)
        cluster_dist = X_cluster_dist[in_cluster]
        cutoff_distance = np.percentile(cluster_dist, percentile_closest)
        above_cutoff = (X_cluster_dist > cutoff_distance)
        X_cluster_dist[in_cluster & above_cutoff] = -1

    partially_propagated = (X_cluster_dist != -1)
    X_train_partially_propagated = X_train[partially_propagated]
    y_train_partially_propagated = y_train[partially_propagated]

    log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=1000, random_state=42)
    log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
    print(len(X_train_partially_propagated))
    print("Trained with representative img labels propagated only to closest instances:", log_reg.score(X_test, y_test))

    print("Accuracy of partially populated labels", np.mean(y_train_partially_propagated == y_train[partially_propagated]))


def draw_silhouette_diagram():
    X, _ = create_blobs()
    kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                    for k in range(1, 10)]
    silhouette_scores = [silhouette_score(X, model.labels_)
                         for model in kmeans_per_k[1:]]

    plt.figure(figsize=(11, 9))

    for k in (3, 4, 5, 6):
        plt.subplot(2, 2, k-2)

        y_pred = kmeans_per_k[k-1].labels_
        silhouette_coefficients = silhouette_samples(X, y_pred)

        padding = len(X)
        pos = padding
        ticks = []
        for i in range(k):
            coeffs = silhouette_coefficients[y_pred == i]
            coeffs.sort()

            color = mpl.cm.Spectral(i / k)
            plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                              facecolor=color, edgecolor=color, alpha=0.7)
            ticks.append(pos + len(coeffs) // 2)
            pos += len(coeffs) + padding

        plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
        plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
        if k in (3, 5):
            plt.ylabel("Cluster")

        if k in (5, 6):
            plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.xlabel("Silhouette Coefficient")
        else:
            plt.tick_params(labelbottom=False)

        plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")
        plt.title("$k={}$".format(k), fontsize=16)

    plt.show()


def kmeans_cluster_count():
    X, _ = create_blobs()
    kmeans_k3 = KMeans(n_clusters=3, random_state=42)
    kmeans_k8 = KMeans(n_clusters=8, random_state=42)
    plot_clusterer_comparison(kmeans_k3, kmeans_k8, X, "$k=3$", "$k=8$")
    plt.show()

    """
    inertia is not a good indicator of optimal number of clusters, because increasing cluster count (k)
    will always decrease inertia. More centroids means that instances will be closer to their centroid.
    """
    print("k=3 inertia", kmeans_k3.inertia_)
    print("k=8 inertia", kmeans_k8.inertia_)


def kmeans_inertia_plot():
    X, _ = create_blobs()
    kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                    for k in range(1, 10)]
    inertias = [model.inertia_ for model in kmeans_per_k]

    plt.figure(figsize=(8, 3.5))
    plt.plot(range(1, 10), inertias, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Inertia", fontsize=14)
    plt.annotate("Elbow",
                 xy=(4, inertias[3]),
                 xytext=(0.55, 0.55),
                 textcoords="figure fraction",
                 fontsize=16,
                 arrowprops=dict(facecolor="black", shrink=0.1)
                 )
    plt.axis([1, 8.5, 0, 1300])
    plt.show()

    plot_decision_boundaries(kmeans_per_k[4-1], X)
    plt.show()


def kmeans_plot_silhouette_score():
    """
    Silhouette score is a better measure than inertia. The plot shows 4 and 5 are both possible good k values.
    It also shows that other values are far worse.
    Silhouette coefficient varies from -1 to 1, where 1 means that the instance is well inside its own cluster and far
    from other clusters, while 0 means it is close to a boundary and -1 means it may be in the wrong cluster.
    :return:
    """
    X, _ = create_blobs()
    kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                    for k in range(1, 10)]
    silhouette_scores = [silhouette_score(X, model.labels_)
                         for model in kmeans_per_k[1:]]

    plt.figure(figsize=(8, 3.5))
    plt.plot(range(2, 10), silhouette_scores, "bo-")
    plt.xlabel("$k", fontsize=14)
    plt.ylabel("Silhouette score", fontsize=14)
    plt.axis([1.8, 8.5, 0.55, 0.7])
    plt.show()


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





def run():
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
    # plot_minibatch_train_times()
    # kmeans_cluster_count()
    # kmeans_inertia_plot()
    # kmeans_plot_silhouette_score()
    # draw_silhouette_diagram()
    # X, y = kmeans_limits()
    # kmeans_draw_limits(X, y)
    # kmeans_img_segmentation()
    # kmeans_for_preprocessing()
    # kmeans_gridsearch()
    kmeans_clustering()

