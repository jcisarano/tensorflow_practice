"""
Some dimensionality reduction techniques can also be used for anomaly detection. For example:
1) take the Olivetti faces dataset and reduce it with PCA, preserving 99% of the variance.
2) Then compute the reconstruction error for each image.
3) Next, take some of the modified images you built in the previous exercise, and look at their reconstruction error:
    notice how much larger the reconstruction error is. If you plot a reconstructed image, you will see why: it tries
    to reconstruct a normal face.
"""
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
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

def run():
    print("anomaly detection")
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_faces_stratified_shuffle()
    X_train_reduced, X_valid_reduced, X_test_reduced = pca_dim_red(X_train, X_valid, X_test)



