import tensorflow as tf
import numpy as np

from custom_loss import load_and_prep_data


def my_softplus(z):
    return tf.math.log(tf.exp(z) + 10)


def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)


def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))


def my_positive_weights(weights):
    return tf.where(weights < 0., tf.zeros_like(weights), weights)


def run():
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = load_and_prep_data()
    input_shape = X_train_scaled.shape[1:]

    layer = tf.keras.layers.Dense(1, activation=my_softplus,
                                  kernel_initializer=my_glorot_initializer,
                                  kernel_regularizer=my_l1_regularizer,
                                  kernel_constraint=my_positive_weights)

    tf.keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(30, activation="selu",
                              kernel_initializer="lecun_normal",
                              input_shape=input_shape),
        tf.keras.layers.Dense(1, activation=my_softplus,
                              kernel_regularizer=my_l1_regularizer,
                              kernel_constraint=my_positive_weights,
                              kernel_initializer=my_glorot_initializer),
    ])

    model.compile(loss="mse", optimizer="nadam", metrics=["mae"])
    model.fit(X_train_scaled, y_train, epochs=2, validation_data=(X_valid_scaled, y_valid), workers=-1)

    save_path = "saved_models/model_w_custom_functions.h5"
    model.save(save_path)

    loaded_model = tf.keras.models.load_model(save_path,
                                              custom_objects={
                                                  "my_l1_regularizer": my_l1_regularizer,
                                                  "my_positive_weights": my_positive_weights,
                                                  "my_glorot_initializer": my_glorot_initializer,
                                                  "my_softplus": my_softplus,
                                              })
    loaded_model.fit(X_train_scaled, y_train, epochs=2, validation_data=(X_valid_scaled, y_valid), workers=-1)

    print("other custom functions")
