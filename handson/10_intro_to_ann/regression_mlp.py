import keras.models
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

def run():
    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

    # StandardScaler will (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html):
    # Standardize features by removing the mean and scaling to unit variance.
    # z = (x - u) / s, where u is the mean and s is the standard deviation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    np.random.seed(42)
    tf.random.set_seed(42)

    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
        keras.layers.Dense(1)
    ])
    model.compile(loss="mean_squared_error",
                  optimizer=keras.optimizers.SGD(learning_rate=1e-3))
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
    mse_test = model.evaluate(X_test, y_test)
    X_new = X_test[:3]
    y_pred = model.predict(X_new)

    # plot history
    plt.plot(pd.DataFrame(history.history))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

    print(y_pred)

