import os

import tensorflow as tf
import numpy as np
import keras
from keras import layers
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CALLBACK_DIR: str = "saved_callbacks"
SAVE_DIR: str = "saved_models"
FILENAME: str = "my_keras_model.h5"

CALLBACK_PATH: str = os.path.join(CALLBACK_DIR, FILENAME)
SAVE_PATH: str = os.path.join(SAVE_DIR, FILENAME)


def run():
    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    model = keras.models.Sequential([
        layers.Dense(30, activation="relu", input_shape=[8]),
        layers.Dense(30, activation="relu"),
        layers.Dense(1)
    ])
    print(model.summary())

    # compile and fit, save checkpoints
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3))
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(CALLBACK_PATH, save_best_only=True)
    history = model.fit(X_train, y_train,
                        epochs=10,
                        validation_data=(X_valid, y_valid),
                        callbacks=[checkpoint_cb],
                        workers=-1)

    # load previously saved model
    model = tf.keras.models.load_model(SAVE_PATH)
    mse_test = model.evaluate(X_test, y_test)
    print("Loaded model MSE:", mse_test)

    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3))
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100,
                        validation_data=[X_valid, y_valid],
                        callbacks=[checkpoint_cb, early_stopping_cb],
                        workers=-1)
    mse_test = model.evaluate(X_test, y_test)
    print("Loaded model:\n", model.summary())
    print("Early stopping MSE:", mse_test)

