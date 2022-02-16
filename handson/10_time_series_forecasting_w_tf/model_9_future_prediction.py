"""
Previous models predicted based on test dataset, but that is not truly future predictions. This model
will make predictions into the future.

"""
import tensorflow as tf
import numpy as np

import utils

BATCH_SIZE: int = 1024
WINDOW_SIZE = 7
HORIZON = 1


def make_prices_windowed(window_size=WINDOW_SIZE, horizon=HORIZON):
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


def create_model(train_dataset):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(HORIZON)
    ], name="model_9_future_prediction")

    model.compile(loss="MAE", optimizer=tf.keras.optimizers.Adam())
    model.fit(train_dataset, epochs=100)


def run():
    train_dataset = make_prices_windowed()
    create_model(train_dataset)


    print("fut pred")


