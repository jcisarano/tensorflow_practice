from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import time


def get_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    mnist.target = mnist.target.astype(np.uint8)

    X = mnist["data"]
    y = mnist["target"]

    return train_test_split(X, y, train_size=60000)


def run():
    X_train, X_test, y_train, y_test = get_mnist()

    print(X_train.shape, X_test.shape)
    rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    t1 = time.time()
    rnd_clf.fit(X_train, y_train)
    t2 = time.time()
    y_pred_rf = rnd_clf.predict(X_test)
    print("Random forest score:", rnd_clf.score(X_test, y_test))
    print("Random forest time:", t2 - t1)
