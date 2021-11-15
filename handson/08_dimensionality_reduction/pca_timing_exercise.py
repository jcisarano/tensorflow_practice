from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np


def get_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    mnist.target = mnist.target.astype(np.uint8)

    X = mnist["data"]
    y = mnist["target"]

    return train_test_split(X, y, train_size=60000)



def run():
    X_train, X_test, y_train, y_test = get_mnist()

    print(X_train.shape, X_test.shape)
    