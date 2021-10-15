# Train a model to get 88%+ accuracy on the fashion MNIST test set. Plot a confusion matrix to see the results after.

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

def run():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    print(x_train.shape, y_train.shape)


