import tensorflow as tf
import numpy as np

from custom_loss import load_and_prep_data, create_huber


def run():
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = load_and_prep_data()
    input_shape = X_train_scaled.shape[1:]

    tf.keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal", input_shape=input_shape),
        tf.keras.layers.Dense(1),
    ])
    model.compile(loss="mse", optimizer="nadam", metrics=[create_huber(2.0)])
    model.fit(X_train_scaled, y_train, epochs=2, workers=-1)


    print("custom metrics")