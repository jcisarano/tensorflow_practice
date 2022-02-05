"""
Same as model 1 but with window size = 30 and horizon size = 7
"""
import os

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers

import utils
from model_1_dense import make_dense_model
from utils import load_data, my_train_test_split, make_windows, make_train_test_splits, create_model_checkpoint

HORIZON: int = 7
WINDOW_SIZE: int = 30


def run():
    timesteps, prices = load_data()
    X_train, X_test, y_train, y_test = my_train_test_split(timesteps, prices)
    full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)
    print(len(full_windows), len(full_labels))
    print(len(train_windows), len(test_windows), len(train_labels), len(test_labels))

    tf.random.set_seed(42)
    model_name = "model_3_dense"
    # model = make_dense_model(model_name, train_windows, test_windows, train_labels, test_labels, output_size=HORIZON)

    # print("Evaluate trained model:")
    # model.evaluate(test_windows, test_labels)

    # load best performing model and evaluate
    model = tf.keras.models.load_model(os.path.join(utils.CHECKPOINT_SAVE_PATH, model_name))

    print("Evaluate best saved model:")
    model.evaluate(test_windows, test_labels)

    preds = utils.make_preds(model, test_windows)
    print(preds[:5])
    print(test_labels[:5])
    results = utils.evaluate_preds(y_true=test_labels, y_pred=preds)
    print(results)

    print("model 3 dense")
