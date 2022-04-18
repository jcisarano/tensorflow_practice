import os.path
import tensorflow as tf

import numpy as np

from time_series import load_data, make_windows, make_train_test_splits

HORIZON: int = 1
WINDOW_SIZE: int = 7


def run():
    timesteps, prices = load_data()
    full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)
    print(len(train_windows), len(train_labels), len(test_windows), len(test_labels))

    tf.random.set_seed(42)

    print("time series cnn")
