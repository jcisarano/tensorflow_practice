"""
The turkey problem is also known as the impact of the highly improbable. The data in the time series changes
unpredictably on a given day due to unknown/unexpected influences.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import utils
from model_9_future_prediction import load_btc_price


def run():
    btc_price = load_btc_price()
    btc_price_turkey = btc_price.copy()

    # change value on one day out of 3000 to show the impact of the highly unlikely
    btc_price_turkey[-1] = btc_price_turkey[-1] * 0.01
    print(btc_price_turkey[-10:])

    bitcoin_prices = utils.load_dataframe()
    btc_timesteps_turkey = np.array(bitcoin_prices.index)
    print(btc_timesteps_turkey[-10:])

    plt.figure(figsize=(10, 7))
    utils.plot_time_series(timesteps=btc_timesteps_turkey, values=btc_price_turkey,
                           format="-", label="BTC Price + Turkey Problem", start=2500)
    plt.show()

    print("the turkey problem")