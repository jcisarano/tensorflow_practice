from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn.model_selection import train_test_split

from utils import plot_time_series, load_data, DATA_PATH, my_train_test_split


def plot_from_x_y(x, y):
    plt.figure(figsize=[10, 7])
    plt.plot(x, y, label="Price", color="#EDA800")
    plt.title(label="Price of Bitcoin from 1 Oct 2013 to 18 May 2021", fontsize=16)
    plt.ylabel("BTC Price", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(fontsize=14)
    axes = plt.gca().axes
    axes.yaxis.grid()
    axes.tick_params(axis="x", labelsize=10)
    axes.tick_params(axis="y", labelsize=10)
    plt.show()


def scatterplot(x0, y0, x1, y1):
    plt.figure(figsize=[10, 7])
    plt.scatter(x0, y0, s=3, label="Test data", color="#EDA800")
    plt.scatter(x1, y1, s=3, label="Train data")
    plt.title(label="Price of Bitcoin from 1 Oct 2013 to 18 May 2021", fontsize=16)
    plt.ylabel("BTC Price", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(fontsize=12)
    axes = plt.gca().axes
    axes.yaxis.grid()
    axes.tick_params(axis="x", labelsize=10)
    axes.tick_params(axis="y", labelsize=10)
    plt.show()


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

    plot_from_x_y(timesteps, btc_price)


def load_csv_1():
    timesteps = []
    btc_price = []
    with open(DATA_PATH, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        next(csv_reader)
        for line in csv_reader:
            timesteps.append(datetime.strptime(line[1], "%Y-%m-%d"))  # date as datetime object
            btc_price.append(float(line[2]))  # price as float

    plot_from_x_y(timesteps, btc_price)


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
    timesteps, prices = load_data()

    # wrong way to split train-test data for time series, because the test data is randomly mixed in with the train data
    # X_train, X_test, y_train, y_test = train_test_split(timesteps, prices, test_size=0.2, random_state=42)
    # print("Wrong way:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # scatterplot(X_train, y_train, X_test, y_test)

    # right way to split train-test data for time series:
    # data stays in chronological order, and last bit is used as test data to represent future (pseudo future data)
    # split_size = int(0.8 * len(prices))  # create 80/20 split
    # X_train, y_train = timesteps[:split_size], prices[:split_size]
    # X_test, y_test = timesteps[split_size:], prices[split_size:]

    X_train, X_test, y_train, y_test = my_train_test_split(timesteps, prices)
    print("Right way:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    scatterplot(X_train, y_train, X_test, y_test)

    # visualize(bitcoin_prices)
    # load_csv()
    # print(prices)
    # load_csv_1()

    plt.figure(figsize=(10, 7))
    plot_time_series(timesteps=X_train, values=y_train, label="Train data")
    plot_time_series(timesteps=X_test, values=y_test, label="Test data")
    plt.show()

    print("bitcoin predict pandas")