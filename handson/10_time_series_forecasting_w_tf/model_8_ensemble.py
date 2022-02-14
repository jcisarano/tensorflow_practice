"""
Ensemble model combines output of multiple *different* models for better decision making
"""

import os
import tensorflow as tf
import utils
from model_7_n_beats import make_datasets, batch_and_prefetch_datasets

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

        model.compile(loss=loss_function, optimizer=tf.keras.optimizers.Adam(), metrics=["mae", "mse"])
        model.fit(train_data, epochs=num_epochs, verbose=0,
                  validation_data=test_data,
                  callbacks=[
                      tf.keras.callbacks.EarlyStopping(
                          monitor="val_loss",
                          patience=200,
                          restore_best_weights=True),
                      tf.keras.callbacks.ReduceLROnPlateau(
                          monitor="val_loss",
                          patience=100,
                          verbose=1)
                  ],
                  workers=-1)
        ensemble_models.append(model)

    return ensemble_models


def run():
    X_train, X_test, y_train, y_test = make_datasets()
    train_dataset, test_dataset = batch_and_prefetch_datasets(X_train, X_test, y_train, y_test)

    ensemble_models = get_ensemble_models(train_dataset, test_dataset, num_iter=5, num_epochs=1000)
    print("ensemble")
