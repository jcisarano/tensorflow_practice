import numpy as np
import tensorflow as tf
import keras.models
from tensorflow.keras import  layers


def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential
    model.add(layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(layers.Dense(n_neurons, activation="relu"))
    model.add(layers.Dense(1))
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)

    return model


def run():
    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    print("yperparameter tuning")

