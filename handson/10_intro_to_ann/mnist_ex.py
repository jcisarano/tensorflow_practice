import os

import keras.backend
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
K = tf.keras.backend


class ExponentialLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.learning_rate))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.learning_rate, self.model.optimizer.learning_rate * self.factor)


def run():
    tf.random.set_seed(42)
    np.random.seed(42)

    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train, X_valid = X_train_full[5000:] / 255., X_train_full[:5000] / 255.
    y_train, y_valid = y_train_full[5000:], y_train_full[:5000]
    X_test = X_test / 255.

    # print("X_train_full shape:", X_train_full.shape, "y_train_full shape:", y_train_full.shape)
    # print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
    # print("X_valid shape:", X_valid.shape, "y_valid shape:", y_valid.shape)
    # print("X_test shape", X_test.shape)#
    # print(X_test[0])

    # visualize some digits
    # n_rows = 4
    # n_cols = 10
    # plt.figure(figsize=(n_cols*1.2, n_rows*1.2))
    # for row in range(n_rows):
    #     for col in range(n_cols):
    #         index = n_cols * row + col
    #         plt.subplot(n_rows, n_cols, index+1)
    #         plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
    #         plt.axis("off")
    #         plt.title(y_train[index], fontsize=12)
    # plt.subplots_adjust(wspace=0.2, hspace=0.5)
    # plt.show()

    tf.keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(300, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
                  metrics=["accuracy"])
    expon_lr = ExponentialLearningRate(factor=1.005)

    history = model.fit(X_train, y_train, epochs=1,
                        validation_data=(X_valid, y_valid),
                        callbacks=[expon_lr],
                        workers=-1)

    plt.plot(expon_lr.rates, expon_lr.losses)
    plt.gca().set_xscale("log")
    plt.hlines(min(expon_lr.losses), min(expon_lr.rates), max(expon_lr.rates))
    plt.axis([min(expon_lr.rates), max(expon_lr.rates), 0, expon_lr.losses[0]])
    plt.grid()
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.show()

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    model = keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(300, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=tf.keras.optimizers.SGD(learning_rate=3e-1),
                  metrics=["accuracy"])
    run_index = 1
    run_logdir = os.path.join("saved_logs", "run_{:03d}".format(run_index))

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=20)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("saved_callbacks/my_mnist_model.h5", save_best_only=True)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

    history = model.fit(X_train, y_train, epochs=100,
                        validation_data=(X_valid, y_valid),
                        callbacks=[early_stopping_cb, checkpoint_cb, tensorboard_cb],
                        workers=-1)



    # print("mnist")
