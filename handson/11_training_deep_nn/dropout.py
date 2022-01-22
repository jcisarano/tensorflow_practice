from functools import partial

import keras.models
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from helper_functions import load_data


class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


class MCAlphaDropout(keras.layers.AlphaDropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


def fit_dropout_model(X_train_scaled, y_train, X_valid_scaled, y_valid):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    n_epochs = 10
    history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                        validation_data=(X_valid_scaled, y_valid),
                        workers=-1)


def fit_alpha_dropout(X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test):
    tf.random.set_seed(42)
    np.random.seed(42)

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.AlphaDropout(rate=0.2),
        keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.AlphaDropout(rate=0.2),
        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.AlphaDropout(rate=0.2),
        keras.layers.Dense(10, activation="softmax")
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    n_epochs = 20
    history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                        validation_data=(X_valid_scaled, y_valid),
                        workers=-1)

    model.evaluate(X_test_scaled, y_test)
    model.evaluate(X_train_scaled, y_train)
    history = model.fit(X_train_scaled, y_train)

    return model


def show_dropout_examples(X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test):
    tf.random.set_seed(42)
    np.random.seed(42)

    model = fit_alpha_dropout(X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test)

    y_probas = np.stack([model(X_test_scaled, training=True) for sample in range(100)])
    y_proba = y_probas.mean(axis=0)
    y_std = y_probas.std(axis=0)

    print(np.round(model.predict(X_test_scaled[:1]), 2))
    print(np.round(y_probas[:, :1], 2))
    print(np.round(y_proba[:1], 2))
    y_std = y_probas.std(axis=0)
    print(np.round(y_std[:1], 2))
    y_pred = np.argmax(y_proba, axis=1)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(accuracy)


def fit_mc_dropout(X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test):
    tf.random.set_seed(42)
    np.random.seed(42)

    model = fit_alpha_dropout(X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test)

    mc_model = keras.models.Sequential([
        MCAlphaDropout(layer.rate) if isinstance(layer, keras.layers.AlphaDropout) else layer
        for layer in model.layers
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    mc_model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    mc_model.set_weights(model.get_weights())
    print(np.round(np.mean([mc_model.predict(X_test_scaled[:1]) for sample in range(100)], axis=0), 2))

    print(mc_model.summary())


def max_norm(X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test):
    layer = keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal",
                               kernel_constraint=keras.constraints.max_norm(1.))
    MaxNormDense = partial(keras.layers.Dense,
                           activation="selu", kernel_initializer="lecun_normal",
                           kernel_constraint=keras.constraints.max_norm(1.))
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        MaxNormDense(300),
        MaxNormDense(100),
        MaxNormDense(10, activation="softmax"),
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

    # fit_dropout_model(X_train_scaled, y_train, X_valid_scaled, y_valid)
    # fit_alpha_dropout(X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test)
    # show_dropout_examples(X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test)
    # fit_mc_dropout(X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test)
    max_norm(X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test)

    print("dropout")


