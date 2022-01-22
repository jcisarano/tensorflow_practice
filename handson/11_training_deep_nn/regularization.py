import keras.layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from helper_functions import load_data


def fit_simple_model(X_train_scaled, y_train, X_valid_scaled, y_valid):
    model = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="elu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dense(100, activation="elu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dense(10, activation="softmax", kernel_regularizer=keras.regularizers.l2(0.01)),
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    n_epochs = 2
    history = model.fit(X_train_scaled, y_train, epochs=n_epochs, validation_data=(X_valid_scaled, y_valid), workers=-1)


def fit_example_w_partial(X_train_scaled, y_train, X_valid_scaled, y_valid):
    from functools import partial
    RegularizedDense = partial(keras.layers.Dense,
                               activation="elu",
                               kernel_initializer="he_normal",
                               kernel_regularizer=keras.regularizers.l2(0.01))
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        RegularizedDense(300),
        RegularizedDense(100),
        RegularizedDense(10, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    n_epochs = 2
    history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                        validation_data=(X_valid_scaled, y_valid),
                        workers=-1)


def run():
    np.random.seed(42)
    tf.random.set_seed(42)

    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()
    pixel_means = X_train.mean(axis=0, keepdims=True)
    pixel_stds = X_train.std(axis=0, keepdims=True)
    X_train_scaled = (X_train - pixel_means) / pixel_stds
    X_valid_scaled = (X_valid - pixel_means) / pixel_stds
    X_test_scaled = (X_test - pixel_means) / pixel_stds

    # to create a layer with l2 regularization
    layer = tf.keras.layers.Dense(100, activation="elu",
                                  kernel_initializer="he_normal",
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01))
    # or, for l1 use keras.regularizers.l1(0.1)
    # or, for l1 AND l2, use keras.regularizers.l1_l2(0.1, 0.01)

    # fit_simple_model(X_train_scaled, y_train, X_valid_scaled, y_valid)
    fit_example_w_partial(X_train_scaled, y_train, X_valid_scaled, y_valid)

    print("regularization")
