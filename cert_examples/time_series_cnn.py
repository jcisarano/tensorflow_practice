import os.path
import tensorflow as tf

import numpy as np

from time_series import load_data, make_windows, make_train_test_splits, CHECKPOINT_PATH, create_model_checkpoint

HORIZON: int = 1
WINDOW_SIZE: int = 7


def make_conv1d_model(train_windows, test_windows, train_labels, test_labels, output_size=HORIZON):
    # input for Conv1D layer must be shape (batch_size, timesteps, input_dim)
    # lambda layer will turn python lambda function into a layer that can be added to model
    # to simplify the data preparation flow
    # In Conv1D layer, filters is num of sliding windows of size kernel & causal padding is good for time sequences

    model_name = "time_series_cnn"
    model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1)),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding="causal", activation="relu"),
        tf.keras.layers.Dense(output_size)
    ], name=model_name)

    model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam())
    model.fit(train_windows, train_labels,
              batch_size=128,
              epochs=100,
              validation_data=(test_windows, test_labels),
              callbacks=[create_model_checkpoint(model_name=model_name,
                                                             save_path=CHECKPOINT_PATH)],
              workers=-1)

    model.evaluate(test_windows, test_labels)


def run():
    timesteps, prices = load_data()
    full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)
    print(len(train_windows), len(train_labels), len(test_windows), len(test_labels))

    tf.random.set_seed(42)

    make_conv1d_model(train_windows, test_windows, train_labels, test_labels)

    print("time series cnn")
