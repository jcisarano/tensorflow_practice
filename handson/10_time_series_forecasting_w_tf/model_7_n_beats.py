"""
Implementation of n-beats algorithm as described in https://arxiv.org/pdf/1905.10437.pdf

From page 3 fig 1:
'Proposed architecture. The basic building block is a multi-layer FC network with RELU
nonlinearities. It predicts basis expansion coefficients both forward, θf, (forecast) and backward, θb,
(backcast). Blocks are organized into stacks using doubly residual stacking principle. A stack may have layers with
shared gb and gf. Forecasts are aggregated in hierarchical fashion. This enables
building a very deep neural network with interpretable outputs.'

 Blocks > Stacks > Deep Network Model

Uses TensorFlow layer subclassing to make custom layers and also Functional API for its custom architecture
"""
import tensorflow as tf

from utils import load_data, load_dataframe

WINDOW_SIZE: int = 7
HORIZON: int = 1
BATCH_SIZE: int = 1024


class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(self,
                 input_size: int, theta_size: int,
                 horizon: int, n_neurons: int,
                 n_layers: int, **kwargs):
        super().__init__(**kwargs)
        self.horizon = horizon
        self.input_size = input_size  # same as window size in previous models
        self.theta_size = theta_size
        self.n_neurons = n_neurons
        self.n_layers = n_layers

        # block contains stack of 4 fully connected layers with ReLU activation
        self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]

        # Output of block is theta layer with linear activation
        self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        theta = self.theta_layer(x)

        backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]

        return backcast, forecast


def test_nbeats_block_class():
    tf.random.set_seed(42)
    # test NBeatsBlock class
    dummy_nbleats_block_layer = NBeatsBlock(input_size=WINDOW_SIZE,
                                            theta_size=WINDOW_SIZE + HORIZON,  # backcast + forecast
                                            horizon=HORIZON,
                                            n_neurons=128,
                                            n_layers=4)

    # create dummy inputs (important that they have the same size as input_size)
    dummy_inputs = tf.expand_dims(tf.range(WINDOW_SIZE) + 1, axis=0)  # input shape to model must reflect input shape
    print(dummy_inputs)
    backcast, forecast = dummy_nbleats_block_layer(dummy_inputs)
    # these are randomized activation outputs of theta layer (no training has happened yet)
    print(f"Backcast: {tf.squeeze(backcast.numpy())}")
    print(f"Forecast: {tf.squeeze(forecast.numpy())}")


def run():
    # test_nbeats_block_class()

    prices = load_dataframe()
    prices_nbeats = prices.copy()
    for i in range(WINDOW_SIZE):
        prices_nbeats[f"Price+{i+1}"] = prices_nbeats["Price"].shift(periods=i+1)
    print(prices_nbeats.head())

    # make features and labels
    X = prices_nbeats.dropna().drop("Price", axis=1)
    y = prices_nbeats.dropna()["Price"]

    # make train and test sets
    split_size = int(len(X) * 0.8)
    X_train, y_train = X[:split_size], y[:split_size]
    X_test, y_test = X[split_size:], y[split_size:]
    print(len(X_train), len(y_train), len(X_test), len(y_test))

    # using tf.data API will make dataset more performant
    # this is more useful for very large datasets
    train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train)
    train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)

    test_features_dataset = tf.data.Dataset.from_tensor_slices(X_test)
    test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_test)

    # combine labels and features to tuple (features, labels)
    train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
    test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset))

    # batch and prefetch
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)  # autotune determines # of available CPUs
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print(train_dataset, test_dataset)

    return 0

    print("n-beats")
