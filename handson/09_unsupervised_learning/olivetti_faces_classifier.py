"""
Continuing with the Olivetti dataset:
1) Train a classifier to predict which person is depicted in each picture and evaluate it on the validation set
2) Next, use K-Means as a dimensionality reduction tool and train a classifier on the reduced set. Search for the
    number of clusters that allows the classifier to get the best performance. What performance can you reach?
3) What if you append the features from the reduced set to the original features (again, searching for the best number
    of clusters)?
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline


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


def train_log_reg(X_train, X_test, y_train, y_test):
    """
    Train and score a LogisticRegression classifier
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
    log_reg.fit(X_train, y_train)

    score = log_reg.score(X_test, y_test)
    print("Log reg baseline score:", score)


def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train and score a RandomForestClassifier
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    rand_forest = RandomForestClassifier(n_estimators=150, random_state=42)
    rand_forest.fit(X_train, y_train)
    score = rand_forest.score(X_test, y_test)
    print("Rand forest baseline score:", score)


def random_forest_kmeans_preprocessing(X_train, X_test, y_train, y_test):
    """
    Try K-Means to preprocess data
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    kmeans = KMeans(n_clusters=115, n_init=10, random_state=42)
    kmeans.fit(X_train)
    X_train_reduced = kmeans.transform(X_train)
    X_test_reduced = kmeans.transform((X_test))

    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train_reduced, y_train)
    score = clf.score(X_test_reduced, y_test)
    print("Rand forest with K-Means dimensionality reduction", score)

    # try it again using pipeline, the results should be the same:
    pipeline = Pipeline([
        ("kmeans", KMeans(n_clusters=120, n_init=10, random_state=42)),
        ("random_forest", RandomForestClassifier(n_estimators=150, random_state=43))
    ])
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print("Rand forest with K-Means preprocessing in pipeline", score)


def random_forest_kmeans_experiments(X_train, X_test, y_train, y_test):
    """
    Use a grid search to try different values for KMeans k score to see which produces the best results
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    pipeline = Pipeline([
        ("k_means", KMeans(n_clusters=10, n_init=10, random_state=42)),
        ("clf", RandomForestClassifier(n_estimators=150, random_state=43))
    ])

    param_grid = dict(k_means__n_clusters=range(10, 150, 5))
    grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
    grid_clf.fit(X_train, y_train)

    print("Grid clf best params:", grid_clf.best_params_)
    print("Best score:", grid_clf.score(X_test, y_test))

    # try it again manually, shouldn't the result be the same?
    for k in range(10, 150, 5):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X_train)
        X_train_dim_red = kmeans.transform(X_train)
        X_test_dim_red = kmeans.transform(X_test)
        clf = RandomForestClassifier(n_estimators=150, random_state=43)
        clf.fit(X_train_dim_red, y_train)
        score = clf.score(X_test_dim_red, y_test)
        print("k={}, score={}".format(k, score))

    # same thing, but using pipeline:
    for k in range(10, 150, 5):
        pipeline = Pipeline([
            ("k_means", KMeans(n_clusters=k, n_init=10, random_state=42)),
            ("clf", RandomForestClassifier(n_estimators=150, random_state=43))
        ])
        pipeline.fit(X_train, y_train)
        print("Pipeline version, k={} and score={}".format(k, pipeline.score(X_test, y_test)))


def kmeans_preprocess_and_append(X_train, X_test, X_validation=None, n_clusters=45):
    """
    Preprocess the data using kmeans, and then append it to the base data to improve the score
    :param X_train:
    :param X_test:
    :param X_validation:
    :param n_clusters:
    :return:
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)

    X_validation_red_dim = None
    X_train_red_dim = kmeans.fit_transform(X_train)
    X_test_red_dim = kmeans.transform(X_test)
    if X_validation is not None:
        X_validation_red_dim = kmeans.transform(X_validation)
        X_validation_red_dim = np.c_[X_validation, X_validation_red_dim]

    X_test_red_dim = np.c_[X_test, X_test_red_dim]
    X_train_red_dim = np.c_[X_train, X_train_red_dim]

    return  X_train_red_dim, X_test_red_dim, X_validation_red_dim


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
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_faces_stratified_shuffle()
    train_log_reg(X_train, X_valid, y_train, y_valid)
    train_random_forest(X_train, X_valid, y_train, y_valid)

    X_train_pca, X_test_pca, X_validation_pca = pca_dim_reduction(X_train, X_test, X_valid)
    print(X_train_pca.shape, X_test_pca.shape, X_validation_pca.shape)
    # train_random_forest(X_train_pca, X_validation_pca, y_train, y_valid)
    # random_forest_kmeans_preprocessing(X_train_pca, X_validation_pca, y_train, y_valid)
    # random_forest_kmeans_experiments(X_train_pca, X_validation_pca, y_train, y_valid)

    X_train_ext, X_test_ext, X_valid_ext = kmeans_preprocess_and_append(X_train, X_test, X_valid)
    train_random_forest(X_train_ext, X_valid_ext, y_train, y_valid)


