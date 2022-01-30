
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

    df.plot(y="Closing Price (USD)")


def run():
    df = pd.read_csv(DATA_PATH,
                     parse_dates=["Date"],  # parses the date column to pandas.datetime
                     index_col=["Date"])  # makes the date column the index, useful because this is sequential data
    visualize(df)

    plt.show()

    print("bitcoin predict pandas")