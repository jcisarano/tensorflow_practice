import os

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

import utils

HORIZON: int = 1
WINDOW_SIZE: int = 7


def make_multivar_model(X_train, X_test, y_train, y_test):
    model_name = "model_6_dense_multivariate"
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(128, activation="relu"),
    #     # tf.keras.layers.Dense(128, activation="relu"),
    #     tf.keras.layers.Dense(HORIZON)
    # ], name=model_name)

    # model.compile(loss="MAE",
    #               optimizer=tf.keras.optimizers.Adam())

    # model.fit(X_train, y_train,
    #           epochs=100,
    #           batch_size=128,
    #           validation_data=(X_test, y_test),
    #           callbacks=[utils.create_model_checkpoint(model_name=model_name, save_path=utils.CHECKPOINT_SAVE_PATH)],
    #           workers=-1)

    # print("Evaluate trained model")
    # model.evaluate(X_test, y_test)

    best_model = tf.keras.models.load_model(os.path.join(utils.CHECKPOINT_SAVE_PATH, model_name))
    print("Evaluate best model:")
    best_model.evaluate(X_test, y_test)

    preds = tf.squeeze(best_model.predict(X_test))
    results = utils.evaluate_preds(y_true=tf.squeeze(y_test), y_pred=preds)
    print(results)


def run():
    X_train, X_test, y_train, y_test = utils.make_windows_multivar()

    tf.random.set_seed(42)
    make_multivar_model(X_train, X_test, y_train, y_test)
