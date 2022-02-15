"""
Ensemble model combines output of multiple *different* models for better decision making
"""

import os
import tensorflow as tf
import utils
import numpy as np
import matplotlib.pyplot as plt
from model_7_n_beats import make_datasets, batch_and_prefetch_datasets

HORIZON: int = 1
WINDOW_SIZE: int = 7


def make_ensemble_preds(ensemble_models, data):
    ensemble_preds = []
    for model in ensemble_models:
        preds = model.predict(data)
        ensemble_preds.append(preds)
    return tf.constant(tf.squeeze(ensemble_preds))


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


def get_upper_lower(preds):
    """
    Prediction ranges can also be useful with ensemble models. E.g. instead of predicting an exact number, use the
    predictions from the models to get a range of confidence values.

    One way to get the 95% confidence prediction intervals for a deep learning model is the bootstrap method:
    1. Take the predictions from a number of randomly initialized models
    2. Measure the standard deviation of the predictions
    3. Multiply the standard deviation by 1.96 (assuming the distribution of data is Gaussian/normal, because that means
        95% of the observations fall within 1.96 standard deviations of the mean)
    4. To get the prediction interval upper and lower bounds, add and subtract the step 3 value to mean of
        predictions made in 1
    * https://en.wikipedia.org/wiki/1.96
    * https://eng.uber.com/neural-networks-uncertainty-estimation/
    """
    std = tf.math.reduce_std(preds, axis=0)
    interval = 1.96 * std
    preds_mean = tf.reduce_mean(preds, axis=0)
    lower, upper = preds_mean - interval, preds_mean + interval

    return lower, upper


def run():
    X_train, X_test, y_train, y_test = make_datasets()
    train_dataset, test_dataset = batch_and_prefetch_datasets(X_train, X_test, y_train, y_test)

    ensemble_models = get_ensemble_models(train_dataset, test_dataset, num_iter=5, num_epochs=1000)

    ensemble_preds = make_ensemble_preds(ensemble_models, test_dataset)

    ensemble_mean = tf.reduce_mean(ensemble_preds, axis=0)
    ensemble_median = np.median(ensemble_preds, axis=0)
    results = utils.evaluate_preds(y_true=y_test, y_pred=ensemble_preds)

    # try reducing median and mean before evaluating, performance seems better with median
    mean_results = utils.evaluate_preds(y_true=y_test, y_pred=ensemble_mean)
    median_results = utils.evaluate_preds(y_true=y_test, y_pred=ensemble_median)
    print("Results", results)
    print("Median Results", median_results)
    print("Mean Results", mean_results)

    lower, upper = get_upper_lower(ensemble_preds)

    # plot median of ensemble preds along with prediction intervals
    offset = 500
    plt.figure(figsize=(10, 7))
    plt.plot(X_test.index[offset:], y_test[offset:], "g", label="Test data")
    plt.plot(X_test.index[offset:], ensemble_median[offset:], "k-", label="Ensemble median")
    plt.xlabel("Date")
    plt.ylabel("BTC price")
    plt.fill_between(X_test.index[offset:], (lower)[offset:], (upper)[offset:], label="Prediction intervals")
    plt.legend(loc="upper left", fontsize=14)
    plt.show()

    print("ensemble")
