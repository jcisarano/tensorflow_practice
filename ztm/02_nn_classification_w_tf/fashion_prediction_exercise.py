# Train a model to get 88%+ accuracy on the fashion MNIST test set. Plot a confusion matrix to see the results after.

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


def softmax(x):
    return tf.exp(x) / tf.sum(tf.exp(x))


def run():
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot", ]
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # print(x_train.shape, y_train.shape)

    x_train_norm = x_train / x_train.max()
    x_test_norm = x_test / x_test.max()
    # print(x_train[0])
    # print(x_train_norm[0])
    y_train_one_hot = tf.one_hot(y_train, depth=10)
    # print(y_train_one_hot.shape)
    # print(y_train_one_hot[0])
    # print(y_train[0] == tf.argmax(y_train_one_hot[0]))

    tf.random.set_seed(42)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(40, activation="relu"),
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    model.fit(x_train_norm, y_train_one_hot, epochs=100, workers=-1)

    y_pred_probabilities = model.predict(x_test_norm)
    y_preds = y_pred_probabilities.argmax(axis=1)

    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_true=y_test, y_pred=y_preds))
    from evaluation import plot_confusion_matrix
    plot_confusion_matrix(y_test, y_preds,
                          classes=class_names, figsize=(15, 15), text_size=10)
