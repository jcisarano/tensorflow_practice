import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

from helper_functions import load_data


def lr_sched_callback(X_train, X_valid, X_test, X_train_scaled, X_valid_scaled,
                      X_test_scaled, y_train, y_valid, y_test):
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax"),
    ])

    optimizer = keras.optimizers.SGD(learning_rate=0.02, momentum=0.9)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    n_epochs = 25
    history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                        validation_data=(X_valid_scaled, y_valid),
                        callbacks=[lr_scheduler],
                        workers=-1)

    plt.plot(history.epoch, history.history["lr"], "bo-")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.tick_params('y', colors='b')
    plt.gca().set_xlim(0, n_epochs-1)
    plt.grid(True)

    ax2 = plt.gca().twinx()
    ax2.plot(history.epoch, history.history["val_loss"], "r^-")
    ax2.set_ylabel("Validation Loss", color="r")
    ax2.tick_params("y", colors="r")

    plt.title("Reduce LR on Plateau", fontsize=14)
    plt.show()


def lr_scheduler(X_train, X_valid, X_test, X_train_scaled, X_valid_scaled,
                      X_test_scaled, y_train, y_valid, y_test):
    np.random.seed(42)
    tf.random.set_seed(42)

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    s = 20 * len(X_train) // 32  # number of steps in 20 epochs where batch size is 32
    learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    n_epochs = 25
    history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                        validation_data=[X_valid_scaled, y_valid],
                        workers=-1)


def run():
    np.random.seed(42)
    tf.random.set_seed(42)

    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()
    pixel_means = X_train.mean(axis=0, keepdims=True)
    pixel_stds = X_train.std(axis=0, keepdims=True)
    X_train_scaled = (X_train - pixel_means) / pixel_stds
    X_valid_scaled = (X_valid - pixel_means) / pixel_stds
    X_test_scaled = (X_test - pixel_means) / pixel_stds

    # lr_sched_callback(X_train, X_valid, X_test, X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test)
    lr_scheduler(X_train, X_valid, X_test, X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test)

    print("perform sched")