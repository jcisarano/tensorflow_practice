import tensorflow as tf

import tensorflow_datasets as tfds


def load_fashion_mnist():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


def load_boston_housing():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data(path="boston_housing.npz",
                                                                                      test_split=0.2, seed=42)


def load_mnist():
    ds = tfds.load("mnist", split="train", shuffle_files=True)
    print(ds)
    return ds


def run():
    load_mnist()
