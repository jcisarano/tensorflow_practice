from sklearn.datasets import fetch_openml
import numpy as np

def get_mnist(m=10000):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)

    idx = np.random.permutation(60000)[:m]

    X = mnist["data"][idx]
    y = mnist["target"][idx]

    return X, y


def run():
    X, y = get_mnist()