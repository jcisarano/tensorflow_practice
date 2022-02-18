"""
The turkey problem is also known as the impact of the highly improbable. The data in the time series changes
unpredictably on a given day due to unknown/unexpected influences.
"""
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import utils
from model_9_future_prediction import load_btc_price

WINDOW_SIZE = 7
HORIZON = 1


def create_model(X_train, X_test, y_train, y_test, epochs=1000, horizon=HORIZON):
    model_name = "model_10_turkey_problem"
    """model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(horizon)
    ], name=model_name)"""
    """model.compile(loss="mae",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["mae"])
    model.fit(x=X_train, y=y_train,
              epochs=epochs,
              validation_data=(X_test, y_test),
              batch_size=128,
              callbacks=[utils.create_model_checkpoint(model.name, utils.CHECKPOINT_SAVE_PATH),
                         tf.keras.callbacks.EarlyStopping(
                             monitor="val_loss",
                             patience=200,
                             restore_best_weights=True
                         ),
                         tf.keras.callbacks.ReduceLROnPlateau(
                             monitor="val_loss",
                             patience=100,
                             verbose=1
                         )],
              workers=-1)
    model.evaluate(X_test, y_test)"""

    loaded_model = tf.keras.models.load_model(os.path.join(utils.CHECKPOINT_SAVE_PATH, model_name))
    print(f"Evaluate {model_name}:")
    loaded_model.evaluate(X_test, y_test)
    preds = utils.make_preds(loaded_model, X_test)

    # preds = model.predict(test_windows)
    results = utils.evaluate_preds(y_true=tf.squeeze(y_test), y_pred=preds)
    print(f"Results for loaded {model_name}:", results)


def plot_turkey(timesteps, prices):
    plt.figure(figsize=(10, 7))
    utils.plot_time_series(timesteps=timesteps, values=prices,
                           format="-", label="BTC Price + Turkey Problem", start=2500)
    plt.show()


def run():
    btc_price = load_btc_price()
    btc_price_turkey = btc_price.copy()

    # change value on one day out of 3000 to show the impact of the highly unlikely
    btc_price_turkey[-1] = btc_price_turkey[-1] * 0.01
    # print(btc_price_turkey[-10:])

    bitcoin_prices = utils.load_dataframe()
    btc_timesteps_turkey = np.array(bitcoin_prices.index)
    # print(btc_timesteps_turkey[-10:])

    # plot_turkey(btc_timesteps_turkey, btc_price_turkey)

    full_windows, full_labels = utils.make_windows(np.array(btc_price_turkey), window_size=WINDOW_SIZE, horizon=HORIZON)
    X_train, X_test, y_train, y_test = utils.make_train_test_splits(full_windows, full_labels)
    # print(len(X_train), len(X_test), len(y_train), len(y_test))

    create_model(X_train, X_test, y_train, y_test)

    print("the turkey problem")
