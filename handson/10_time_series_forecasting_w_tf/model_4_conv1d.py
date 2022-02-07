"""
# Conv1D model parts:
    # lambda layer
    # Conv1D model, filters=128, kernel_size=?, padding=?
    # output layer = dense

"""
import tensorflow as tf
from utils import load_data, my_train_test_split, make_windows, make_train_test_splits

HORIZON: int = 1
WINDOW_SIZE: int = 7


def run():
    timesteps, prices = load_data()
    X_train, X_test, y_train, y_test = my_train_test_split(timesteps, prices)
    full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
    print(len(full_windows), len(full_labels))
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)
    print(len(train_windows), len(train_labels), len(test_windows), len(test_labels))

    tf.random.set_seed(42)
    model_name = "model_1_dense"

    # input for Conv1D layer must be shape (batch_size, timesteps, input_dim)
    print(train_windows[0].shape)
    # so we must reshape it to fit our needs
    x = tf.constant(train_windows[0])
    expand_dims_layer = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))
    # lambda layer will turn python lambda function into a layer, and you can add it to the model to simplify
    # the data preparation flow
    print(f"Original shape: {x.shape}")
    print(f"Expanded shape: {expand_dims_layer(x).shape}")  # adds extra dimension
    print(f"Original values with expanded shape  {expand_dims_layer(x)}")


