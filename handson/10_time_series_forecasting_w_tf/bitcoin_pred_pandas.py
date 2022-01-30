from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

DATA_PATH: str = "data/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv"


def load_csv():
    timesteps = []
    btc_price = []
    with open(DATA_PATH, mode="r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)
        data = list(csv_reader)
    for item in data:
        timesteps.append(datetime.strptime(item[1], "%Y-%m-%d"))
        btc_price.append(float(item[2]))

    plt.figure(figsize=[10, 7])
    plt.plot(timesteps, btc_price, label="Price", color="#EDA800")
    plt.title(label="Price of Bitcoin from 1 Oct 2013 to 18 May 2021", fontsize=16)
    plt.ylabel("BTC Price", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(fontsize=14)
    axes = plt.gca().axes
    axes.yaxis.grid()
    axes.tick_params(axis="x", labelsize=10)
    axes.tick_params(axis="y", labelsize=10)
    plt.show()



def load_csv_1():
    timesteps = []
    btc_price = []
    with open(DATA_PATH, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        next(csv_reader)
        for line in csv_reader:
            timesteps.append(datetime.strptime(line[1], "%Y-%m-%d"))  # date as datetime object
            btc_price.append(float(line[2]))  # price as float

    plt.figure(figsize=[10, 7])
    plt.plot(timesteps, btc_price, label="Price", color="#EDA800")
    plt.title(label="Price of Bitcoin from 1 Oct 2013 to 18 May 2021", fontsize=16)
    plt.ylabel("BTC Price", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(fontsize=14)
    axes = plt.gca().axes
    axes.yaxis.grid()
    axes.tick_params(axis="x", labelsize=10)
    axes.tick_params(axis="y", labelsize=10)
    plt.show()


def visualize(df):
    print(df.head())  # first few entries
    print(df.tail())  # last few entries
    print(df.info())  # data types
    print(len(df))  # how many samples
    # only about 8 years of daily data, so not many entries - small datasets common with time series data
    # "seasonality" of time series datasets is the number of samples per year. For this Bitcoin data, the seasonality
    # is daily, so only 365 samples per year

    df.plot(title="Price of Bitcoin from 1 Oct 2013 to 18 May 2021", fontsize=16, figsize=[10, 7])
    plt.ylabel("BTC Price", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(fontsize=14)
    axes = plt.gca().axes
    axes.yaxis.grid()
    axes.tick_params(axis="x", labelsize=10)
    axes.tick_params(axis="y", labelsize=10)
    plt.show()


def run():
    # df = pd.read_csv(DATA_PATH,
    #                  parse_dates=["Date"],  # parses the date column to pandas.datetime
    #                  index_col=["Date"])  # makes the date column the index, useful because this is sequential data

    # bitcoin_prices = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})
    # visualize(bitcoin_prices)
    load_csv()
    #print(prices)
    load_csv_1()

    print("bitcoin predict pandas")