"""
First experiment is a simple dense model:
    * A single dense layer with 128 hidden units and ReLU
    * An output layer with linear activation (no activation)
    * Adam optimizer and MAE loss function
    * Batch size 128 (larger because data rows are smaller, only 7 items per row)
    * 100 epochs
"""
import tensorflow as tf
from tensorflow.keras import layers

from utils import load_data, my_train_test_split, make_windows, make_train_test_splits, HORIZON, WINDOW_SIZE, \
    create_model_checkpoint


def run():
    timesteps, prices = load_data()
    X_train, X_test, y_train, y_test = my_train_test_split(timesteps, prices)
    full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(HORIZON, activation="linear")  # linear activation outputs the value passed in from the dense layer
    ], name="model_1_dense")

    model.compile(loss="mae",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["mae", "mse"])

    model.fit(x=train_windows,
              y=train_labels,
              epochs=100,
              verbose=1,
              batch_size=128,
              validation_data=(test_windows, test_labels),
              callbacks=[create_model_checkpoint(model_name=model.name)],
              workers=-1
    )

    print("dense model")
