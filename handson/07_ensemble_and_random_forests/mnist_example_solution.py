import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier


def get_data():
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
    mnist.target = mnist.target.astype(np.int8)  # converts from string to int
    X_train_val, X_test, y_train_val, y_test = train_test_split(mnist.data,
                                                                mnist.target,
                                                                test_size=10000,
                                                                random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=10000, random_state=42)

    return X_train, X_test, X_val, y_train, y_val, y_test


def run():
    X_train, X_test, X_val, y_train, y_val, y_test = get_data()
    print(X_train.shape, X_test.shape)

