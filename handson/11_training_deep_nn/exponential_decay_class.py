import tensorflow as tf
import numpy as np
from tensorflow import keras
import keras.callbacks

from helper_functions import load_data
from model_reuse import load_split_data

K = keras.backend


# custom class allows updating learning rate at each iteration rather than at each epoch (see learning_rate_scheduler)
class ExponentialDecay(keras.callbacks.Callback):
    def __init__(self, s=40000):
        super().__init__()
        self.s = s

    def on_batch_begin(self, batch, logs=None):
        # Note: the "batch" arg is reset at each epoch
        lr = K.get_value(self.model.optimizer.learning_rate)
        K.set_value(self.model.optimizer.learning_rate, lr * 0.1 ** (1 / self.s))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.learning_rate)


def run():
    np.random.seed(42)
    tf.random.set_seed(42)

    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()
    pixel_means = X_train.mean(axis=0, keepdims=True)
    pixel_stds = X_train.std(axis=0, keepdims=True)
    X_train_scaled = (X_train - pixel_means) / pixel_stds
    X_valid_scaled = (X_valid - pixel_means) / pixel_stds
    X_test_scaled = (X_test - pixel_means) / pixel_stds

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="relu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(200, activation="relu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax"),
    ])

    lr0 = 0.01
    optimizer = tf.keras.optimizers.Nadam(learning_rate=lr0)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])

    n_epochs = 25
    s = 20*len(X_train) // 32  # num steps in 20 epochs (where batch size is 32)
    exp_decay = ExponentialDecay(s)
    history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                        validation_data=(X_valid_scaled, y_valid),
                        callbacks=[exp_decay],
                        workers=-1)

    print("ed class")
