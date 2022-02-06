"""

"""
import tensorflow as tf
from utils import load_data, my_train_test_split, make_windows, make_train_test_splits

HORIZON: int = 1
WINDOW_SIZE: int = 7

def run():
    timesteps, prices = load_data()
    X_train, X_test, y_train, y_test = my_train_test_split(timesteps, prices)
    full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

    tf.random.set_seed(42)
    model_name = "model_1_dense"