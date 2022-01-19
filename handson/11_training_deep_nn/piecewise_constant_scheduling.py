import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

from helper_functions import load_data


def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001


def piecewise_constant(boundaries, values):
    boundaries = np.array([0] + boundaries)
    values = np.array(values)

    def piecewise_constant_fn(epoch):
        return values[np.argmax(boundaries > epoch) - 1]

    return piecewise_constant_fn


def run():
    np.random.seed(42)
    tf.random.set_seed(42)

    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()
    pixel_means = X_train.mean(axis=0, keepdims=True)
    pixel_stds = X_train.std(axis=0, keepdims=True)
    X_train_scaled = (X_train - pixel_means) / pixel_stds
    X_valid_scaled = (X_valid - pixel_means) / pixel_stds
    X_test_scaled = (X_test - pixel_means) / pixel_stds

    piecewise_constant_fn = piecewise_constant([5, 15], [0.01, 0.005, 0.001])
    lr_scheduler = keras.callbacks.LearningRateScheduler(piecewise_constant_fn)
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="nadam",
                  metrics=["accuracy"])
    n_epochs = 25
    history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                        validation_data=[X_valid_scaled, y_valid],
                        callbacks=[lr_scheduler],
                        workers=-1)

    plt.plot(history.epoch, [piecewise_constant_fn(epoch) for epoch in history.epoch], "o-")
    plt.axis([0, n_epochs - 1, 0, 0.011])
    plt.grid(True)
    plt.title("Piecewise Constant Scheduling")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.show()

    print("pcs")
