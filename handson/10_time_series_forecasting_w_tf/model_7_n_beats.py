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

WINDOW_SIZE: int = 7
HORIZON: int = 1

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
    test_nbeats_block_class()

    return 0

    print("n-beats")
