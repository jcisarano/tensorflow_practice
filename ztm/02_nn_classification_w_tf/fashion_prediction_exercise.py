# Train a model to get 88%+ accuracy on the fashion MNIST test set. Plot a confusion matrix to see the results after.

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


def softmax(x):
    return tf.exp(x) / tf.sum(tf.exp(x))


def run():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    print(x_train.shape, y_train.shape)

    x_train_norm = x_train / x_train.max()
    print(x_train[0])
    print(x_train_norm[0])
    y_train_one_hot = tf.one_hot(y_train, depth=10)
    # print(y_train_one_hot.shape)
    # print(y_train_one_hot[0])
    # print(y_train[0] == tf.argmax(y_train_one_hot[0]))

    tf.random.set_seed(42)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dense(5, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    model.fit(x_train_norm, y_train_one_hot, epochs=100, workers=-1)

