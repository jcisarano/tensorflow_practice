"""
Ensemble model combines output of multiple *different* models for better decision making
"""

import os
import tensorflow as tf
import utils

HORIZON: int = 1
WINDOW_SIZE: int = 7


def get_ensemble_models(train_data, test_data,
                        horizon=HORIZON, num_iter=10,
                        num_epochs=1000,
                        loss_fns=["mae", "mse", "mape"]):
    """
    Returns a list of num_iter models, each trained on MAE, MSE, and MAPE loss
        E.g.: for num_iter=10, a list of 30 trained models is returned, num_iter * len(loss_fns)

    :param train_data:
    :param test_data:
    :param horizon:
    :param num_iter:
    :param num_epochs:
    :param loss_fn:
    :return:
    """
    ensemble_models = []
    for i in range(num_iter):
        for loss_function in loss_fns:
            print(f"Optimizing model by reducing: {loss_function} for {num_epochs} epochs, model number {i}")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, kernel_initializer="he_normal", activation="relu"),
            tf.keras.layers.Dense(128, kernel_initializer="he_normal", activation="relu"),
            tf.keras.layers.Dense(horizon),
        ])



def run():
    print("ensemble")