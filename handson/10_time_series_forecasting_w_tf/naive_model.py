# Naive model: predict next timestep as last timestep:
# $$\hat{y}_{t} = y_{t-1}$$
# The prediction at timestep t (y-hat) is equal to the value at timestep t-1 (previous timestep) - this is
# for horizon of one

import matplotlib.pyplot as plt
import tensorflow as tf

from utils import plot_time_series, load_data, my_train_test_split, mean_absolute_scaled_error, evaluate_preds, \
    get_labelled_window

HORIZON = 1
WINDOW_SIZE = 7


def naive_forecast(y):
    return y[:-1]


def run():
    timesteps, prices = load_data()
    X_train, X_test, y_train, y_test = my_train_test_split(timesteps, prices)

    n_forecast = naive_forecast(y_test)
    plt.figure(figsize=[10, 7])
    # plot_time_series(timesteps=X_train, values=y_train, label="Train data")
    plot_time_series(timesteps=X_test, values=y_test, start=350, format="-", label="Test data")
    plot_time_series(timesteps=X_test[1:], values=n_forecast, start=350, format="-", label="Naive forecast data")
    plt.show()

    # Check MASE implementation. This output should be very close to one:
    # print("MASE", mean_absolute_scaled_error(y_true=y_test[1:], y_pred=n_forecast))

    results = evaluate_preds(y_true=y_test[1:], y_pred=n_forecast)
    print(results)

    # test window label function
    test_window, test_label = get_labelled_window(tf.expand_dims(tf.range(8), axis=0), horizon=HORIZON)
    print(f"Window: {tf.squeeze(test_window).numpy()} -> Label: {tf.squeeze(test_label).numpy()}")
