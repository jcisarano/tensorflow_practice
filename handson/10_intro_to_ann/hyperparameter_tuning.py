import numpy as np
import tensorflow as tf
import keras.models
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import  layers


def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(layers.Dense(n_neurons, activation="relu"))
    model.add(layers.Dense(1))
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)

    return model


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
    X_new = X_test[:3]

    keras_reg = tf.keras.wrappers.scikit_learn.KerasRegressor(build_model)
    keras_reg.fit(X_train, y_train, epochs=200,
                  validation_data=(X_valid, y_valid),
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)],
                  workers=-1)
    mse_test = keras_reg.score(X_test, y_test)
    print("MSE:", mse_test)
    y_pred = keras_reg.predict(X_new)
    print(y_pred)

    print("yperparameter tuning")

