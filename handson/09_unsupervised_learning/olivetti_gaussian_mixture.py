"""
1) Train a Gaussian mixture model on the Olivetti faces dataset. To speed up the algorithm, you should probably reduce
    the dataset's dimensionality (e.g., use PCA, preserving 99% of the variance).
2) Use the model to generate some new faces (using the sample() method), and visualize them (if you used PCA,
    you will need to use its inverse_transform() method).
3) Try to modify some images (e.g., rotate, flip, darken) and see if the model can detect the anomalies (i.e., compare
    the output of the score_samples() method for normal images and for anomalies).
"""
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


def load_faces():
    (X, y) = fetch_olivetti_faces(return_X_y=True)
    # stratification is done by default
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    return X_train, X_test, y_train, y_test


def load_faces_stratified_shuffle():
    """
    Use a different method to load and split data into train/validation/test sets
    :return:
    """
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


def pca_dim_reduction(X_train, X_test, X_validation=None):
    """
    Use PCA to reduce dimensionality and improve training speed
    :param X_train:
    :param X_test:
    :param X_validation:
    :return:
    """
    # 0.99 retains 99% of data variance
    pca = PCA(0.99)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    X_validation_pca = None
    if X_validation is not None:
        X_validation_pca = pca.transform(X_validation)

    return X_train_pca, X_test_pca, X_validation_pca


def run():
    print("oh so gaussian!")