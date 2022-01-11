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
LOG_DIR: str = "saved_logs"
FILENAME: str = "my_keras_model.h5"

CALLBACK_PATH: str = os.path.join(CALLBACK_DIR, FILENAME)
SAVE_PATH: str = os.path.join(SAVE_DIR, FILENAME)


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(LOG_DIR, run_id)


def run():
    run_logdir = get_run_logdir()
    print(run_logdir)

    tf.keras.backend.clear_session()
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
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3))

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(CALLBACK_PATH, save_best_only=True)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
    history = model.fit(X_train, y_train, epochs=30,
                        validation_data=(X_valid, y_valid),
                        callbacks=[checkpoint_cb, tensorboard_cb],
                        workers=-1)

    print("tensorboard")
