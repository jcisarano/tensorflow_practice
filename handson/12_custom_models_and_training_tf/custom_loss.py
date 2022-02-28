"""

"""
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def load_and_prep_data():
    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target.reshape(-1, 1), random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test


def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5

    return tf.where(is_small_error, squared_loss, linear_loss)


def plot_loss_function():
    plt.figure(figsize=(8, 3.5))
    z = np.linspace(-4, 4, 200)
    plt.plot(z, huber_fn(0, z), "b-", linewidth=2, label="huber($z$)")
    plt.plot(z, z ** 2 / 2, "b:", linewidth=1, label=r"$\frac{1}{2}z^2$")
    plt.plot([-1, -1], [0, huber_fn(0., -1.)], "r--")
    plt.plot([1, 1], [0, huber_fn(0., 1.)], "r--")
    plt.gca().axhline(y=0, color='k')
    plt.gca().axvline(x=0, color='k')
    plt.axis([-4, 4, 0, 4])
    plt.grid(True)
    plt.xlabel("$z$")
    plt.legend(fontsize=14)
    plt.title("Huber loss", fontsize=14)
    plt.show()


def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss = threshold * tf.abs(error) - threshold ** 2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

    return huber_fn


class HuberLoss(tf.keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold ** 2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


def run():
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = load_and_prep_data()

    # plot_loss_function()

    input_shape = X_train_scaled.shape[1:]
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal", input_shape=input_shape),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss=huber_fn, optimizer="nadam", metrics=["mae"])
    model.fit(X_train_scaled, y_train, epochs=2, validation_data=(X_valid_scaled, y_valid), workers=-1)

    save_path = "saved_models/model_w_custom_loss.h5"
    model.save(save_path)

    print("\nLoaded model\n")
    loaded_model = tf.keras.models.load_model(save_path,
                                              custom_objects={"huber_fn": huber_fn})
    loaded_model.fit(X_train_scaled, y_train, epochs=2, validation_data=(X_valid_scaled, y_valid), workers=-1)

    print("\nCompile and fit with custom loss threshold\n")
    loaded_model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=["mae"])
    loaded_model.fit(X_train_scaled, y_train, epochs=2,
                     validation_data=(X_valid_scaled, y_valid))

    save_path_custom = "saved_models/model_w_custom_loss_threshold.h5"
    loaded_model.save(save_path_custom)

    print("\nLoaded model with custom threshold\n")
    # load model and define loss function with custom threshold
    loaded_model_cust = tf.keras.models.load_model(save_path_custom,
                                                   custom_objects={"huber_fn": create_huber(2.0)})

    loaded_model_cust.fit(X_train_scaled, y_train, epochs=2,
                          validation_data=(X_valid_scaled, y_valid))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal", input_shape=input_shape),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss=HuberLoss(2.), optimizer="nadam", metrics=["mae"])
    print("\nFit model with HuberLoss class\n")
    model.fit(X_train_scaled, y_train, epochs=2,
              validation_data=(X_valid_scaled, y_valid),
              workers=-1)

    save_path_class = "saved_models/model_w_custom_loss_class.h5"
    model.save(save_path_class)
    loaded_model = tf.keras.models.load_model(save_path_class, custom_objects={"HuberLoss":HuberLoss})
    print("\nFit loaded model with HuberLoss class\n")
    loaded_model.fit(X_train_scaled, y_train, epochs=2,
                     validation_data=(X_valid_scaled, y_valid),
                     workers=-1)
    print("\nLoaded model threshold:", loaded_model.loss.threshold)

    print("custom loss")
