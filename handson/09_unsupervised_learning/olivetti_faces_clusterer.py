"""
Exercise: The classic Olivetti faces dataset contains 400 grayscale 64 × 64–pixel images of faces. Each image is
flattened to a 1D vector of size 4,096. 40 different people were photographed (10 times each), and the usual task is to
train a model that can predict which person is represented in each picture.

1) Load the dataset using the sklearn.datasets.fetch_olivetti_faces() function.
2) then split it into a training set, a validation set, and a test set (note that the dataset is already scaled between
    (0 and 1). Since the dataset is quite small, you probably want to use stratified sampling to ensure that there are
    the same number of images per person in each set.
3) Next, cluster the images using K-Means and ensure that you have a good number of clusters (using one of the techniques
    discussed in this chapter).
4) Visualize the clusters. Do you see the same faces in each cluster?
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


def load_faces():
    (X, y) = fetch_olivetti_faces(return_X_y=True)
    # stratification is done by default
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    return X_train, X_test, y_train, y_test


def load_faces_stratified_shuffle():
    olivetti = fetch_olivetti_faces()
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=40, random_state=42)
    train_valid_idx, test_idx = next(strat_split.split(olivetti.data, olivetti.target))
    x_train_valid = olivetti.data[train_valid_idx]
    y_train_valid = olivetti.target[train_valid_idx]
    X_test = olivetti.data[test_idx]
    y_test = olivetti.data[test_idx]

    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=80, random_state=43)
    train_idx, valid_idx = next(strat_split.split(x_train_valid, y_train_valid))
    X_train = x_train_valid[train_idx]
    y_train = y_train_valid[train_idx]
    X_valid = x_train_valid[valid_idx]
    y_valid = y_train_valid[valid_idx]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def pca_dimensionality_reduction(X_train, X_test, X_valid=None):
    """
    Dimensionality reduction will speed up the training
    :param X_train:
    :param X_test:
    :param X_valid:
    :return:
    """
    pca = PCA(0.99)
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)

    if X_valid is not None:
        X_valid = pca.fit_transform(X_valid)

    return X_train, X_valid, X_test


def kmeans_cluster_experiment(X, y):
    k_range = range(10, 150, 5)
    kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                    for k in k_range]
    scores = [silhouette_score(X, model.labels_)
              for model in kmeans_per_k]

    best_idx = np.argmax(scores)
    best_kval = k_range[best_idx]
    best_score = scores[best_idx]

    # plot silhouette graph
    plt.figure(figsize=(8, 3.5))
    plt.plot(k_range, scores, "bo-")
    plt.plot(best_kval, best_score, "ro")
    plt.xlabel("k", fontsize=14)
    plt.ylabel("Silhouette score", fontsize=14)
    # plt.axis([1, 150, 0.55, 0.7])
    plt.show()

    print("Best k by silhouette score:", best_kval)



def train_kmeans(X, y, n_clusters=10, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    print(kmeans.labels_)

    return kmeans


def visualize_images(kmeans, images, labels, k=10):
    y_pred = kmeans.predict(images)

    n_cols = 10
    count = 0
    plt.figure(figsize=(10, 12))
    for ii in range(kmeans.n_clusters):
        for idx, img in enumerate(images):
            if y_pred[idx] != ii:
                continue
            plt.subplot(len(images) // n_cols, n_cols, count+1)
            plt.subplots_adjust(top=0.99, bottom=0.01, left=0.1, right=0.90)
            plt.imshow(img.reshape(64, 64), cmap="gray")
            plt.axis("off")
            plt.title("Cluster {}({})".format(ii, labels[idx]), fontsize=8)
            count = count+1

    #n_cols = 10
    #plt.figure(figsize=(10, 10))
    #for idx, img in enumerate(images):
    #    plt.subplot(len(images) // n_cols, n_cols, idx+1)
    #    plt.imshow(img.reshape(64, 64), cmap="gray")
    #    plt.axis("off")

    plt.show()


def run():
    X_train, X_test, y_train, y_test = load_faces()
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_faces_stratified_shuffle()
    X_train_pca, X_valid_pca, X_test_pca = pca_dimensionality_reduction(X_train=X_train, X_valid=X_valid, X_test=X_test)


    kmeans_cluster_experiment(X_train_pca, y_train)

    #kmeans = train_kmeans(X_train, y_train, n_clusters=50)

    #print(kmeans)

    #visualize_images(kmeans, X_test, y_test)

