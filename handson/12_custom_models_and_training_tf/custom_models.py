import tensorflow as tf
import numpy as np
import keras

from custom_loss import load_and_prep_data


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [tf.keras.layers.Dense(n_neurons, activation="elu",
                                             kernel_initializer="he_normal")
                       for _ in range(n_layers)]

    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
            return inputs + Z


class ResidualRegressor(tf.keras.models.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = tf.keras.layers.Dense(30, activation="elu",
                                             kernel_initializer="he_normal")
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = tf.keras.layers.Dense(output_dim)

    def call(self, inputs):
        Z = self.hidden1(inputs)
        for _ in range(1 + 3):
            Z = self.block1(Z)
        Z = self.block2(Z)
        return self.out(Z)


class ReconstructingRegressor(tf.keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [tf.keras.layers.Dense(30, activation="selu",
                                             kernel_initializer="lecun_normal")
                       for _ in range(5)]
        self.out = tf.keras.layers.Dense(output_dim)
        self.reconstruction_mean = tf.keras.metrics.Mean(name="reconstruction_error")

    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = tf.keras.layers.Dense(n_inputs)

    def call(self, inputs, training=None):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction-inputs))
        self.add_loss(0.05*recon_loss)
        if training:
            result = self.reconstruction_mean(recon_loss)
            self.add_metric(result)
        return self.out(Z)




def run():
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = load_and_prep_data()
    X_new_scaled = X_test_scaled

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    model = ResidualRegressor(1)
    model.compile(loss="mse", optimizer="nadam")
    history = model.fit(X_train_scaled, y_train, epochs=5, workers=-1)
    score = model.evaluate(X_test_scaled, y_test)
    y_pred = model.predict(X_new_scaled)

    model_save_path="saved_models/custom_model.ckpt"
    model.save(model_save_path)

    loaded_model = tf.keras.models.load_model(model_save_path)
    history = loaded_model.fit(X_train_scaled, y_train, epochs=5, workers=-1)

    # same idea, uses Sequential API

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    block1 = ResidualBlock(2, 30)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal"),
        block1, block1, block1,
        ResidualBlock(2, 30),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss="mse", optimizer="nadam")
    model.fit(X_train_scaled, y_train, epochs=5, workers=-1)
    score = model.evaluate(X_test_scaled, y_test)
    y_pred = model.predict(X_new_scaled)


    print("custom models")
