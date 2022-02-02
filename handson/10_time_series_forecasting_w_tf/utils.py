from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf

DATA_PATH: str = "data/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv"


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


def load_data(data_path=DATA_PATH):
    df = pd.read_csv(data_path,
                     parse_dates=["Date"],  # parses the date column to pandas.datetime
                     index_col=["Date"])  # makes the date column the index, useful because this is sequential data

    bitcoin_prices = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})
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

    return {"mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.numpy(),
            "mase": mase.numpy()}
