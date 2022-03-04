import keras.backend
import tensorflow as tf
import numpy as np

from custom_loss import load_and_prep_data


class MyDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, batch_input_shape):
        self.kernel = self.add_weight(
            name="kernel", shape=[batch_input_shape[-1], self.units],
            initializer="glorot_normal"
        )
        self.bias = self.add_weight(
            name="bias", shape=[self.units], initializer="zeros"
        )
        super().build(batch_input_shape)  # must be at end

    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)

    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1]+[self.units])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units,
                "activation": tf.keras.activations.serialize(self.activation)}


def cust_dense_layer_class(X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test, input_shape):
    model = tf.keras.models.Sequential([
        MyDense(30, activation="relu", input_shape=input_shape),
        MyDense(1),
    ])
    model.compile(loss="mse", optimizer="nadam")
    model.fit(X_train_scaled, y_train, epochs=2,
              validation_data=(X_valid_scaled, y_valid))
    model.evaluate(X_test_scaled, y_test)


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

    # cust_exp_layer(X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test, input_shape)
    cust_dense_layer_class(X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test, input_shape)

    print("custom layers")
