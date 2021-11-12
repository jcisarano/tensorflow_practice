"""
One way to choose the number of dimensions is to calculate the variance as dimensions are reduced, and keep the
variance above 95%.

But if you are reducing dimensions for visualization, you'll have to go to just 2 or 3 dims.
"""
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def get_mnist_train_test_split():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    mnist.target = mnist.target.astype(np.uint8)

    X = mnist["data"]
    y = mnist["target"]

    return train_test_split(X, y)


def run():
    X_train, X_test, y_train, y_test = get_mnist_train_test_split()

    # create the PCA, then fit it, then determine number of dims to keep variance above 95%
    pca = PCA()
    pca.fit(X_train)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    # print(cumsum.shape)
    d = np.argmax(cumsum >= 0.95) + 1
    print(d)

