"""
# Conv1D model parts:
    # lambda layer
    # Conv1D model, filters=128, kernel_size=?, padding=?
    # output layer = dense

"""
import tensorflow as tf

import utils
from utils import load_data, my_train_test_split, make_windows, make_train_test_splits

HORIZON: int = 1
WINDOW_SIZE: int = 7


def make_conv1d_model(train_windows, test_windows, train_labels, test_labels, output_size=HORIZON):
    # input for Conv1D layer must be shape (batch_size, timesteps, input_dim)
    # lambda layer will turn python lambda function into a layer, and you can add it to
    # the model to simplify the data preparation flow
    # In Conv1D layer, filters is num of sliding windows of size kernel & causal padding is good for time sequences

    model_name = "model_4_conv1d"
    model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1)),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding="causal", activation="relu"),
        tf.keras.layers.Dense(output_size)
    ], name=model_name)

    model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam())
    model.fit(train_windows, train_labels,
              batch_size=128,
              epochs=100,
              validation_data=(test_windows, test_labels),
              callbacks=[utils.create_model_checkpoint(model.name, utils.CHECKPOINT_SAVE_PATH)],
              workers=-1)


def run():
    timesteps, prices = load_data()
    X_train, X_test, y_train, y_test = my_train_test_split(timesteps, prices)
    full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
    print(len(full_windows), len(full_labels))
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)
    print(len(train_windows), len(train_labels), len(test_windows), len(test_labels))

    tf.random.set_seed(42)

    make_conv1d_model(train_windows, test_windows, train_labels, test_labels)



