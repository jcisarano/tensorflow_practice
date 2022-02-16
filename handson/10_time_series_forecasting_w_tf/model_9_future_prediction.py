"""
Previous models predicted based on test dataset, but that is not truly future predictions. This model
will make predictions into the future.

"""
import tensorflow as tf
import numpy as np

import utils

BATCH_SIZE: int = 1024

def make_prices_windowed(window_size=7, horizon=1):
    bitcoin_prices_windowed = utils.create_block_reward_date_ranges()
    for i in range(window_size):
        bitcoin_prices_windowed[f"Price+{i+1}"] = bitcoin_prices_windowed["Price"].shift(periods=i+1)

    # print(bitcoin_prices_windowed.head)

    X_all = bitcoin_prices_windowed.dropna().drop(["Price", "block_reward"], axis=1).astype(np.float32)
    y_all = bitcoin_prices_windowed.dropna()["Price"].astype(np.float32)

    # print(X.head)
    # print(y.head)

    features_dataset_all = tf.data.Dataset.from_tensor_slices(X_all)
    labels_dataset_all = tf.data.Dataset.from_tensor_slices(y_all)

    dataset_all = tf.data.Dataset.zip((features_dataset_all, labels_dataset_all))

    # batch and prefetch for optimal performance
    dataset_all = dataset_all.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # print(dataset_all)

    return dataset_all

    # split_size = int(len(X_all) * 0.8)
    # X_train, y_train = X_all[:split_size], y_all[:split_size]
    # X_test, y_test = X_all[split_size:], y_all[split_size:]
    # # print(len(X_train), len(y_train), len(X_test), len(y_test))

    # return X_train, X_test, y_train, y_test


def run():
    X_train, X_test, y_train, y_test = make_prices_windowed()


    print("fut pred")


