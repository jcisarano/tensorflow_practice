
from tensorflow.keras.datasets import fashion_mnist

import tensorflow_datasets as tfds


def load_fashion_mnist():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


def load_mnist():
    ds = tfds.load("mnist", split="train", shuffle_files=True)
    print(ds)
    return ds


def run():
    load_mnist()
