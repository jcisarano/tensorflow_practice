import keras.models
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=[8]),
        keras.layers.Dense(30, activation="relu"),
        keras.layers.Dense(1)
    ])

    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3))
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
    mse_test = model.evaluate(X_test, y_test)
    print(mse_test)
    model.save("saved_models/my_keras_model.h5")

    model = keras.models.load_model("saved_models/my_keras_model.h5")
    print(model.predict(X_new))

    model.save_weights("saved_models/my_keras_weights.ckpt")
    model.load_weights("saved_models/my_keras_weights.ckpt")

