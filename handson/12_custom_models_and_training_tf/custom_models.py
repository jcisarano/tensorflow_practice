import tensorflow as tf
import numpy as np
import keras

from custom_loss import load_and_prep_data


class ResidualBlock(keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(n_neurons, activation="elu",
                                          kernel_initializer="he_normal")
                       for _ in range(n_layers)]

    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
            return inputs + Z


def run():
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = load_and_prep_data()
    input_shape = X_train_scaled.shape[1:]
    X_new_scaled = X_test_scaled

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)



    print("custom models")
