import math
import matplotlib.pyplot as plt
from tensorflow import keras

from helper_functions import load_data


def run():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()
    pixel_means = X_train.mean(axis=0, keepdims=True)
    pixel_stds = X_train.std(axis=0, keepdims=True)
    X_train_scaled = (X_train - pixel_means) / pixel_stds
    X_valid_scaled = (X_valid - pixel_means) / pixel_stds
    X_test_scaled = (X_test - pixel_means) / pixel_stds

    # SGD optimizer with decay:
    optimizer = keras.optimizers.SGD(learning_rate=0.01, decay=1e-4)

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    n_epochs = 25
    history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                        validation_data=(X_valid_scaled, y_valid),
                        workers=-1)



    print("lr scheduling")

