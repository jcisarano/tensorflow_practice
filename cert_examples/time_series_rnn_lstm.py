import tensorflow as tf
from time_series import load_data, make_windows, make_train_test_splits

HORIZON: int = 1
WINDOW_SIZE: int = 7


def make_lstm_model(model_name, train_windows, test_windows, train_labels, test_labels, output_size):
    print("make model")


def run():
    timesteps, prices = load_data()
    full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

    tf.random.set_seed(42)
    model_name = "time_series_rnn_lstm"
    make_lstm_model(model_name, train_windows, test_windows, train_labels, test_labels, HORIZON)

    print("time series rnn")
