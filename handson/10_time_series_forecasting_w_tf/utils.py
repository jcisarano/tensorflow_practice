from matplotlib import pyplot as plt
import pandas as pd

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