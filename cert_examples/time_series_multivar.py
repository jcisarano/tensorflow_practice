import os.path
import tensorflow as tf

import numpy as np
import pandas as pd

import time_series
from time_series import load_dataframe

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


def create_block_reward_data_ranges():
    path = os.path.join(time_series.DATA_PATH, time_series.FILENAME)
    bitcoin_prices = load_dataframe(path)
    block_reward_2_days = (block_reward_3_datetime - bitcoin_prices.index[0]).days
    block_reward_3_days = (block_reward_4_datetime - bitcoin_prices.index[0]).days
    print(block_reward_2_days, block_reward_3_days)

    bitcoin_prices_block = bitcoin_prices.copy()
    bitcoin_prices_block["block_reward"] = None

    bitcoin_prices_block.iloc[:block_reward_2_days, -1] = block_reward_3
    bitcoin_prices_block.iloc[block_reward_2_days:block_reward_3_days, -1] = block_reward_3
    bitcoin_prices_block.iloc[block_reward_3_days:, -1] = block_reward_4

    print(bitcoin_prices_block.head())
    print(bitcoin_prices_block.iloc[1500:1505])
    print(bitcoin_prices_block.tail())

    return bitcoin_prices_block


def make_windows_multivar(window_size=7, horizon=1):
    bitcoin_prices_windowed = create_block_reward_data_ranges()


def run():
    X_train, X_test, y_train, y_test = make_windows_multivar()


    print("multivariate time series")
