import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd


def run():
    np.random.seed(42)
    tf.random.set_seed(42)
    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    X_new = X_test[:3]

    #
    input_ = keras.layers.Input(shape=X_train.shape[1:])
    hidden1 = keras.layers.Dense(30, activation="relu")(input_)
    hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
    concat = keras.layers.concatenate([input_, hidden2])
    output = keras.layers.Dense(1)(concat)
    model = tf.keras.Model(inputs=[input_], outputs=[output])
    print(model.summary())

    model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(learning_rate=1e-3))
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid), workers=-1)
    mse_test = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_new)
    print("MSE TEST:", mse_test)
    print("Pred new:", y_pred)

    plt.plot(pd.DataFrame(history.history))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

    # #####
    # different subsets of input features
    np.random.seed(42)
    tf.random.set_seed(42)

    input_A = keras.layers.Input(shape=[5], name="wide_output")
    input_B = keras.layers.Input(shape=[6], name="deep_input")
    hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
    hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
    concat = keras.layers.concatenate([input_A, hidden2])
    output = keras.layers.Dense(1, name="output")(concat)
    model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])

    model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))
    X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
    X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
    X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
    X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

    history = model.fit((X_train_A, X_train_B), y_train, epochs=20,
                        validation_data=((X_valid_A, X_valid_B), y_valid))
    mse_test = model.evaluate((X_test_A, X_test_B), y_test)
    y_pred = model.predict((X_new_A, X_new_B))
    print("MSE TEST subset model:", mse_test)
    print("y_pred subset model:", y_pred)

