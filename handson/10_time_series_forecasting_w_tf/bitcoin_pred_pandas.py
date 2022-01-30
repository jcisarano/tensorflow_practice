
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH: str = "data/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv"


def visualize(df):
    print(df.head())  # first few entries
    print(df.tail())  # last few entries
    print(df.info())  # data types
    print(len(df))  # how many samples
    # only about 8 years of daily data, so not many entries - small datasets common with time series data
    # "seasonality" of time series datasets is the number of samples per year. For this Bitcoin data, the seasonality
    # is daily, so only 365 samples per year

    df.plot(title="Price of Bitcoin from 1 Oct 2013 to 18 May 2021", fontsize=16, figsize=[10,7])
    plt.ylabel("BTC Price", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(fontsize=14)
    axes = plt.gca().axes
    axes.yaxis.grid()
    # plt.setp(axes.get_xticklabels(), fontsize=10)
    axes.tick_params(axis="x", labelsize=10)
    axes.tick_params(axis="y", labelsize=10)
    plt.show()


def run():
    df = pd.read_csv(DATA_PATH,
                     parse_dates=["Date"],  # parses the date column to pandas.datetime
                     index_col=["Date"])  # makes the date column the index, useful because this is sequential data

    bitcoin_prices = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})
    visualize(bitcoin_prices)

    print("bitcoin predict pandas")