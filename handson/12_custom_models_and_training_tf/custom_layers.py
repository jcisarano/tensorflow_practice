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


class MyMultiLayer(tf.keras.layers.Layer):
    def call(self, X):
        X1, X2 = X
        print(f"X1.shape: {X1.shape} X2.shape: {X2.shape}")
        return X1+X2, X1*X2

    def compute_output_shape(self, batch_input_shape):
        batch_input_shape1, batch_input_shape2 = batch_input_shape
        return [batch_input_shape1, batch_input_shape2]


def split_data(data):
    columns_count = data.shape[-1]
    half = columns_count // 2
    return data[:, :half], data[:, half:]


def multilayer_test(X_train_scaled, X_valid_scaled, X_test_scaled):
    inputs1 = tf.keras.layers.Input(shape=[2])
    inputs2 = tf.keras.layers.Input(shape=[2])
    outputs1, outputs2 = MyMultiLayer()((inputs1, inputs2))

    X_train_scaled_A, X_train_scaled_B = split_data(X_train_scaled)
    X_valid_scaled_A, X_valid_scaled_B = split_data(X_valid_scaled)
    X_test_scaled_A, X_test_scaled_B = split_data(X_test_scaled)

    print("\nSplit data:\n")
    print(X_train_scaled_A.shape, X_train_scaled_B.shape)

    outputs1, outputs2 = MyMultiLayer()((X_train_scaled_A, X_train_scaled_B))


def cust_dense_layer_class(X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test, input_shape):
    model = tf.keras.models.Sequential([
        MyDense(30, activation="relu", input_shape=input_shape),
        MyDense(1),
    ])
    model.compile(loss="mse", optimizer="nadam")
    model.fit(X_train_scaled, y_train, epochs=2,
              validation_data=(X_valid_scaled, y_valid))
    model.evaluate(X_test_scaled, y_test)

    save_path = "saved_models/model_w_custom_layer.h5"
    model.save(save_path)
    loaded_model = tf.keras.models.load_model(save_path,
                                              custom_objects={"MyDense": MyDense})
    print("\nEvaluate loaded model:\n")
    loaded_model.evaluate(X_test_scaled, y_test)


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
    # cust_dense_layer_class(X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test, input_shape)

    multilayer_test(X_train_scaled, X_valid_scaled, X_test_scaled)

    print("custom layers")
