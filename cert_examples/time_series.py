import os.path
import tensorflow as tf

import numpy as np
import pandas as pd

DATA_PATH: str = "datasets/time_series"
FILENAME: str = "BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv"

HORIZON: int = 1
WINDOW_SIZE: int = 7

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
    split_size = int(split*len(y))
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
    window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
    window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=1)
    windowed_array = x[window_indexes]
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)

    return windows, labels


def run():
    timesteps, prices = load_data()
    # X_train, X_test, y_train, y_test = my_train_test_split(timesteps, prices)
    full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)

    print("time series")
