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

    return X_test_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test


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
    plt.plot(z, z**2/2, "b:", linewidth=1, label=r"$\frac{1}{2}z^2$")
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


def run():
    X_test_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = load_and_prep_data()

    plot_loss_function()

    print("custom loss")