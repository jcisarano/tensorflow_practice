import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
import os

DATA_PATH: str = "data/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv"
CHECKPOINT_SAVE_PATH: str = "model_experiments"

# bitcoin block reward halving events
block_reward_1 = 50     # 3 Jan 2009
block_reward_2 = 25     # 8 Nov 2012
block_reward_3 = 12.5   # 9 July 2016
block_reward_4 = 6.25   # 18 May 2020

# halving event dates
block_reward_1_datetime = np.datetime64("2009-01-03")
block_reward_2_datetime = np.datetime64("2012-11-08")
block_reward_3_datetime = np.datetime64("2016-07-09")
block_reward_4_datetime = np.datetime64("2020-05-18")


def create_block_reward_date_ranges():
    bitcoin_prices = load_dataframe()
    block_reward_2_days = (block_reward_3_datetime - bitcoin_prices.index[0]).days
    block_reward_3_days = (block_reward_4_datetime - bitcoin_prices.index[0]).days
    # print(block_reward_2_days, block_reward_3_days)

    bitcoin_prices_block = bitcoin_prices.copy()
    bitcoin_prices_block["block_reward"] = None

    bitcoin_prices_block.iloc[:block_reward_2_days, -1] = block_reward_2
    bitcoin_prices_block.iloc[block_reward_2_days:block_reward_3_days, -1] = block_reward_3
    bitcoin_prices_block.iloc[block_reward_3_days:, -1] = block_reward_4

    # print(bitcoin_prices_block.head())
    # print(bitcoin_prices_block.iloc[1500:1505])
    # print(bitcoin_prices_block.tail())

    return bitcoin_prices_block


def make_windows_multivar(window_size=7, horizon=1):
    bitcoin_prices_windowed = create_block_reward_date_ranges()
    for i in range(window_size):
        bitcoin_prices_windowed[f"Price+{i+1}"] = bitcoin_prices_windowed["Price"].shift(periods=i+1)

    # print(bitcoin_prices_windowed.head)

    X = bitcoin_prices_windowed.dropna().drop("Price", axis=1).astype(np.float32)
    y = bitcoin_prices_windowed.dropna()["Price"].astype(np.float32)

    # print(X.head)
    # print(y.head)

    split_size = int(len(X) * 0.8)
    X_train, y_train = X[:split_size], y[:split_size]
    X_test, y_test = X[split_size:], y[split_size:]
    # print(len(X_train), len(y_train), len(X_test), len(y_test))

    return X_train, X_test, y_train, y_test


def plot_time_series(timesteps, values, format=".", start=0, end=None, label=None):
    """
    Plots series of points in times against values
    :param timesteps: array of timestep values
    :param values: array of values across time
    :param format: style of plot, default = .
    :param start: where to start the plot, value is used as index of timesteps
    :param end: where to end the plot, index for timesteps
    :param label: label to give to plot values
    :return:
    """
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("BTC price")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


def my_train_test_split(X, y, split=0.8):
    split_size = int(split * len(y))
    X_train, y_train = X[:split_size], y[:split_size]
    X_test, y_test = X[split_size:], y[split_size:]
    return X_train, X_test, y_train, y_test


def make_train_test_splits(windows, labels, test_split=0.2):
    split_size = int(len(windows) * (1-test_split))
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]

    return train_windows, test_windows, train_labels, test_labels


def load_dataframe(data_path=DATA_PATH):
    df = pd.read_csv(data_path,
                     parse_dates=["Date"],  # parses the date column to pandas.datetime
                     index_col=["Date"])  # makes the date column the index, useful because this is sequential data

    bitcoin_prices = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})
    return bitcoin_prices


def load_data(data_path=DATA_PATH):
    bitcoin_prices = load_dataframe(data_path=data_path)
    timesteps = bitcoin_prices.index.to_numpy()
    prices = bitcoin_prices["Price"].to_numpy()

    return timesteps, prices


def mean_absolute_scaled_error(y_true, y_pred):
    """
    Implement MASE assuming no seasonality of the data
    :param y_true:
    :param y_pred:
    :return:
    """
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))  # shift left 1 for seasonality of one day

    return mae / mae_naive_no_season


def evaluate_preds(y_true, y_pred):
    # make sure everything is float32
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)

    # Account for different sized metrics. For longer horizons, reduce to single value
    if mae.ndim > 0:
        mae = tf.reduce_mean(mae)
        mse = tf.reduce_mean(mse)
        rmse = tf.reduce_mean(rmse)
        mape = tf.reduce_mean(mape)
        mase = tf.reduce_mean(mase)

    return {"mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.numpy(),
            "mase": mase.numpy()}


# function to label windowed data
def get_labelled_window(x, horizon):
    """
    Creates labels for windowed dataset
    e.g. if horizon = 1,
    Input: [0, 1, 2, 3, 4, 5, 6, 7] -> Output: ([0, 1, 2, 3, 4, 5, 6], [7])
    :param x:
    :param horizon:
    :return:
    """

    return x[:, :-horizon], x[:, -horizon:]


def make_windows(x, window_size, horizon):
    """
    Turns 1d array into 2d array of sequential labelled windows of window_size with horizon sized labels
    Should be the same result as keras function tf.keras.utils.timeseries_dataset_from_array
    :param x:
    :param window_size:
    :param horizon:
    :return:
    """

    # create the window, len win_size + horizon (includes the data and label at this point)
    window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
    # creates 2d array of indices of size (data length, window_step length)
    # window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T
    window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=1)
    # print(window_indexes.shape)
    # convert array of indices to temp array of actual data
    windowed_array = x[window_indexes]
    # split temp windows into windows array and labels array
    windows, labels = get_labelled_window(windowed_array, horizon=horizon)

    return windows, labels


def create_model_checkpoint(model_name, save_path=CHECKPOINT_SAVE_PATH):
    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name),
                                              verbose=0,
                                              save_best_only=True)


def make_preds(model, input_data):
    """
    Uses model to make predictions on input data
    :param model:
    :param input_data: Should be windowed as when training model
    :return:
    """
    forecast = model.predict(input_data)
    return tf.squeeze(forecast)  # squeeze to return 1d array of preds

