import keras.backend
import tensorflow as tf
import numpy as np

from custom_loss import load_and_prep_data


def cust_exp_layer(X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test, input_shape):
    exponential_layer = tf.keras.layers.Lambda(lambda x: tf.exp(x))
    print(exponential_layer([-1., 0., 1.]))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(30, activation="relu", input_shape=input_shape),
        tf.keras.layers.Dense(1),
        exponential_layer
    ])

    model.compile(loss="mse", optimizer="sgd")
    model.fit(X_train_scaled, y_train, epochs=5,
              validation_data=(X_valid_scaled, y_valid),
              workers=-1)
    model.evaluate(X_test_scaled, y_test)


def run():
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = load_and_prep_data()
    input_shape = X_train_scaled.shape[1:]

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    cust_exp_layer(X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test, input_shape)

    print("custom layers")
