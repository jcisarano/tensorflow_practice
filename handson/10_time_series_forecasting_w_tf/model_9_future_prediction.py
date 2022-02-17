"""
Previous models predicted based on test dataset, but that is not truly future predictions. This model
will make predictions into the future.

For time series forecasts, you have to retrain the model every time you want to make a prediction. See:
https://towardsdatascience.com/3-facts-about-time-series-forecasting-that-surprise-experienced-machine-learning-practitioners-69c18ee89387

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import utils

BATCH_SIZE: int = 1024
WINDOW_SIZE = 7
HORIZON = 1

INTO_FUTURE = 14


def make_prices_windowed(window_size=WINDOW_SIZE, horizon=HORIZON):
    bitcoin_prices_windowed = utils.create_block_reward_date_ranges()
    for i in range(window_size):
        bitcoin_prices_windowed[f"Price+{i+1}"] = bitcoin_prices_windowed["Price"].shift(periods=i+1)

    # print(bitcoin_prices_windowed.head)
    print(bitcoin_prices_windowed.tail())

    X_all = bitcoin_prices_windowed.dropna().drop(["Price", "block_reward"], axis=1).astype(np.float32)
    y_all = bitcoin_prices_windowed.dropna()["Price"].to_numpy()

    # print(X.head)
    # print(y.head)

    features_dataset_all = tf.data.Dataset.from_tensor_slices(X_all)
    labels_dataset_all = tf.data.Dataset.from_tensor_slices(y_all)

    dataset_all = tf.data.Dataset.zip((features_dataset_all, labels_dataset_all))

    # batch and prefetch for optimal performance
    dataset_all = dataset_all.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # print(dataset_all)

    return dataset_all, X_all, y_all


def create_model(train_dataset, X_all, y_all):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(HORIZON)
    ], name="model_9_future_prediction")

    model.compile(loss="MAE", optimizer=tf.keras.optimizers.Adam())
    model.fit(train_dataset, epochs=100)

    future_forecast = make_future_forecast(historical_dataset=y_all, model=model)

    return future_forecast


def make_future_forecast(historical_dataset, model, into_future=INTO_FUTURE, window_size=WINDOW_SIZE) -> list:
    """
    Make future forecasts into_future steps after values end.
    Returns future forecasts as a list of floats
    """
    future_forecasts = []
    last_window = historical_dataset[-window_size:]

    for _ in range(into_future):
        future_pred = model.predict(tf.expand_dims(last_window, axis=0))
        print(f"Predicting on:\n{last_window} -> Prediction: {tf.squeeze(future_pred).numpy()}\n")
        future_forecasts.append(tf.squeeze(future_pred).numpy())
        last_window = np.append(last_window, future_pred)[-window_size:]

    return future_forecasts


def run():
    train_dataset, X_all, y_all = make_prices_windowed()
    create_model(train_dataset, X_all, y_all)


    print("fut pred")


