import os.path

import pandas as pd

DATA_PATH: str = "datasets/time_series"
FILENAME: str = "BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv"


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


def run():
    timesteps, prices = load_data()
    print("time series")
