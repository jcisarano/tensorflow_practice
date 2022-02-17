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

    print("the turkey problem")