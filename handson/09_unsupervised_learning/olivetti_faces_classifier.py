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
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


def load_faces():
    (X, y) = fetch_olivetti_faces(return_X_y=True)
    # stratification is done by default
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    return X_train, X_test, y_train, y_test


def train_log_reg(X_train, X_test, y_train, y_test):
    log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
    log_reg.fit(X_train, y_train)

    score = log_reg.score(X_test, y_test)
    print("Log reg baseline score:", score)


def train_random_forest(X_train, X_test, y_train, y_test):
    rand_forest = RandomForestClassifier(n_estimators=150, random_state=42)
    rand_forest.fit(X_train, y_train)
    score = rand_forest.score(X_test, y_test)
    print("Rand forest baseline score:", score)


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


def pca_dim_reduction(X_train, X_test, X_validation=None):
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
    train_random_forest(X_train_pca, X_validation_pca, y_train, y_valid)

