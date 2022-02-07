"""

"""

import tensorflow as tf

import utils
from utils import load_data, my_train_test_split, make_windows, make_train_test_splits

HORIZON: int = 1
WINDOW_SIZE: int = 7


def make_lstm_model(model_name, train_windows, test_windows, train_labels, test_labels, output_size=HORIZON):
    inputs = tf.keras.layers.Input(shape=WINDOW_SIZE)
    x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs)
    # x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(128, activation="relu")(x)
    # x = tf.keras.layers.Dense(32, activation="relu")(x)
    output = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=output, name=model_name)
    model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam())
    model.fit(train_windows, train_labels,
              epochs=100, verbose=1,
              batch_size=128,
              validation_data=(test_windows, test_labels),
              callbacks=[utils.create_model_checkpoint(model_name=model_name, save_path=utils.CHECKPOINT_SAVE_PATH)],
              workers=1)

    return 0


def run():
    timesteps, prices = load_data()
    X_train, X_test, y_train, y_test = my_train_test_split(timesteps, prices)
    full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

    tf.random.set_seed(42)
    model_name = "model_5_lstm"
    results = make_lstm_model(model_name, train_windows, test_windows, train_labels, test_labels)
    return results
