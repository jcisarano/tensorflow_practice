# Naive model: predict next timestep as last timestep:
# $$\hat{y}_{t} = y_{t-1}$$
# The prediction at timestep t (y-hat) is equal to the value at timestep t-1 (previous timestep) - this is
# for horizon of one

import matplotlib.pyplot as plt

from utils import plot_time_series, load_data, my_train_test_split


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
