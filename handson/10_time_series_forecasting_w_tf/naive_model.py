# Naive model: predict next timestep as last timestep:
# $$\hat{y}_{t} = y_{t-1}$$
# The prediction at timestep t (y-hat) is equal to the value at timestep t-1 (previous timestep) - this is
# for horizon of one
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import utils
from utils import plot_time_series, load_data, my_train_test_split, mean_absolute_scaled_error, evaluate_preds, \
    get_labelled_window, make_windows, make_train_test_splits

HORIZON = 1
WINDOW_SIZE = 7


def naive_forecast(y):
    return y[:-1]


def run():
    timesteps, prices = load_data()
    X_train, X_test, y_train, y_test = my_train_test_split(timesteps, prices)

    n_forecast = naive_forecast(y_test)
    # plt.figure(figsize=[10, 7])
    # plot_time_series(timesteps=X_train, values=y_train, label="Train data")
    # plot_time_series(timesteps=X_test, values=y_test, start=350, format="-", label="Test data")
    # plot_time_series(timesteps=X_test[1:], values=n_forecast, start=350, format="-", label="Naive forecast data")
    # plt.show()

    # Check MASE implementation. This output should be very close to one:
    # print("MASE", mean_absolute_scaled_error(y_true=y_test[1:], y_pred=n_forecast))

    results = evaluate_preds(y_true=y_test[1:], y_pred=n_forecast)
    print(results)

    # test window label function
    # test_window, test_label = get_labelled_window(tf.expand_dims(tf.range(8), axis=0), horizon=HORIZON)
    # print(f"Window: {tf.squeeze(test_window).numpy()} -> Label: {tf.squeeze(test_label).numpy()}")

    # test full window label function
    # full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
    # print(len(full_windows), len(full_labels))

    # for i in range(3):
    #     print(f"Window: {full_windows[i]} -> Label: {full_labels[i]}")

    # for i in range(3):
    #     print(f"Window: {full_windows[i - 3]} -> Label: {full_labels[i - 3]}")

    # test create train and test windows
    # train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)
    # print(len(train_windows), len(test_windows))

    # verify that the labels are the same before and after the split
    # print(np.array_equal(np.squeeze(train_labels[:-HORIZON-1]), y_train[WINDOW_SIZE:]))

    return results

