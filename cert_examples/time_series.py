import os.path
import tensorflow as tf

import numpy as np
import pandas as pd

DATA_PATH: str = "datasets/time_series"
FILENAME: str = "BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv"

HORIZON: int = 1  # horizon size can change to change preds
WINDOW_SIZE: int = 7  # window size can change to increase/decrease amount of data per label

CHECKPOINT_PATH: str = "models/time_series_checkpoints/"


def load_dataframe(data_path):
    df = pd.read_csv(data_path,
                     parse_dates=["Date"],  # parses date col to pandas.datetime
                     index_col=["Date"])  # makes date col the index, important for sequential data
    prices = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})
    return prices


def load_data():
    path = os.path.join(DATA_PATH, FILENAME)
    raw_prices = load_dataframe(data_path=path)
    timesteps = raw_prices.index.to_numpy()
    prices = raw_prices["Price"].to_numpy()

    return timesteps, prices


def my_train_test_split(X, y, split=0.8):
    split_size = int(split * len(y))
    X_train, y_train = X[:split_size], y[:split_size]
    X_test, y_test = X[split_size:], y[split_size:]

    return X_train, X_test, y_train, y_test


def get_labelled_windows(x, horizon):
    return x[:, :-horizon], x[:, -horizon:]


def make_windows(x, window_size, horizon):
    """
    Turns 1d array into 2d array of sequential labelled windows of window_size with horizon sized labels
    :param x:
    :param window_size:
    :param horizon:
    :return:
    """
    window_step = np.expand_dims(np.arange(window_size + horizon), axis=0)
    window_indexes = window_step + np.expand_dims(np.arange(len(x) - (window_size + horizon - 1)), axis=1)
    windowed_array = x[window_indexes]
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)

    return windows, labels


def make_train_test_splits(windows, labels, test_split=0.2):
    split_size = int(len(windows) * (1 - test_split))
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]

    return train_windows, test_windows, train_labels, test_labels


def create_model_checkpoint(model_name, save_path=CHECKPOINT_PATH):
    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name),
                                              verbose=0,
                                              save_best_only=True)


def make_dense_model(model_name, train_windows, test_windows, train_labels, test_labels, output_size=HORIZON):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(output_size, activation="linear")
        ], name=model_name
    )
    model.compile(loss="mae",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["mae"])
    model.fit(x=train_windows,
              y=train_labels,
              epochs=100,
              verbose=1,
              batch_size=128,
              validation_data=(test_windows, test_labels),
              callbacks=[create_model_checkpoint(model_name=model.name)],
              workers=-1)

    return model


def run():
    timesteps, prices = load_data()
    # X_train, X_test, y_train, y_test = my_train_test_split(timesteps, prices)
    full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

    tf.random.set_seed(42)
    model = make_dense_model("dense_model", train_windows, test_windows, train_labels, test_labels)

    print("time series")
