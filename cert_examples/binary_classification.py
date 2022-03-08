"""
Common ways to improve model performance:
    Adding layers
    Increase the number of hidden units in the layers
    Change the activation functions of the layers
    Change the optimization function of the model
    Change the learning rate of the optimization function
    Fit on more data
    Fit for longer
"""

import tensorflow as tf

from utils import generate_circles


def simple_linear(X, y):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    model.fit(X, y, epochs=100, workers=-1)


def binary_classification_nonlinear(X, y):
    """
    Simple binary classification good for nonlinear data
    :param X:
    :param y:
    :return:
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(4, activation="relu"),  # nonlinear activation function
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    model.fit(X, y, epochs=50, workers=-1)


def run():
    X, y = generate_circles()

    tf.random.set_seed(42)
    binary_classification_nonlinear(X, y)

    print("binary classification")
