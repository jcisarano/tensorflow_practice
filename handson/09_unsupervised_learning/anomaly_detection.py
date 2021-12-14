"""
Some dimensionality reduction techniques can also be used for anomaly detection. For example:
1) take the Olivetti faces dataset and reduce it with PCA, preserving 99% of the variance.
2) Then compute the reconstruction error for each image.
3) Next, take some of the modified images you built in the previous exercise, and look at their reconstruction error:
    notice how much larger the reconstruction error is. If you plot a reconstructed image, you will see why: it tries
    to reconstruct a normal face.
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit


def load_faces_stratified_shuffle():
    faces = fetch_olivetti_faces()
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=40, random_state=42)
    train_valid_idx, test_idx = next(strat_split.split(faces.data, faces.target))
    x_train_valid = faces.data[train_valid_idx]
    y_train_valid = faces.target[train_valid_idx]
    X_test = faces.data[test_idx]
    y_test = faces.target[test_idx]

    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=40, random_state=43)
    train_idx, valid_idx = next(strat_split.split(x_train_valid, y_train_valid))
    X_train = faces.data[train_idx]
    y_train = faces.target[train_idx]
    X_valid = faces.data[valid_idx]
    y_valid = faces.data[valid_idx]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def pca_dim_red(X_train, X_valid, X_test):
    """
    Dimensionality reduction using PCA
    :param X_train:
    :param X_valid:
    :param X_test:
    :return:
    """
    # 0.99 means that 99% of variance is retained
    pca = PCA(0.99)
    X_train_reduced = pca.fit_transform(X_train)
    X_valid_reduced = pca.transform(X_valid)
    X_test_reduced = pca.transform(X_test)

    return X_train_reduced, X_valid_reduced, X_test_reduced


def pca_reconstruction_error(X_train, X_train_reduced):
    pca = PCA(0.99)
    pca.fit(X_train)

    X_train_reconstruct = pca.inverse_transform(X_train_reduced)

    X_train_mse = mean_squared_error(X_train, X_train_reconstruct)

    print("X_train reconstruction error:", X_train_mse)

    # calc mse manually, result is the same:
    X_train_mse = np.square(X_train_reconstruct - X_train).mean(axis=-1)
    print("X_train reconstruction error manual:", X_train_mse.mean())

    return X_train_reconstruct


def modify_faces(X, y):
    n_rotated = 4
    rotated = np.transpose(X[:n_rotated].reshape(-1, 64, 64), axes=[0, 2, 1])
    rotated = rotated.reshape(-1, 64*64)
    y_rotated = y[:n_rotated]

    n_flipped = 3
    flipped = X[:n_flipped].reshape(-1, 64, 64)[:, ::-1]
    flipped = flipped.reshape(-1, 64*64)
    y_flipped = y[:n_flipped]

    n_darkened = 3
    darkened = X[:n_darkened].copy()
    darkened[:, 1:-1] *= 0.3
    y_darkened = y[:n_darkened]

    X_mod = np.r_[rotated, flipped, darkened]
    y_mod = np.concatenate([y_rotated, y_flipped, y_darkened])

    return X_mod, y_mod


def plot_faces(X, y, n_cols=5):
    plt.figure(figsize=(8, 5))

    for idx, img in enumerate(X):
        plt.subplot(len(X)//n_cols, n_cols, idx+1)
        plt.imshow(img.reshape(64, 64), cmap="gray")
        plt.title(y[idx])
        plt.axis("off")
    plt.show()


def run():
    print("anomaly detection")
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_faces_stratified_shuffle()
    X_train_reduced, X_valid_reduced, X_test_reduced = pca_dim_red(X_train, X_valid, X_test)

    pca_reconstruction_error(X_train, X_train_reduced)

    X_mod, y_mod = modify_faces(X_train, y_train)
    plot_faces(X_mod, y_mod)




