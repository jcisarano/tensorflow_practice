import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np


def plot_multiple_images(images, labels, class_names, predictions=None, pred_probs=None):
    plt.figure(figsize=(10, 9))
    for i in range(25):
        index = np.random.randint(len(images))
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        # plt.grid(false)
        plt.imshow(images[index], cmap=plt.cm.binary)
        color = "black"
        if predictions is not None:
            label = "{} ({:.0f}%)".format(class_names[predictions[index]], np.amax(pred_probs[index])*100)
            if predictions[index] != labels[index]:
                color = "red"
        else:
            label = class_names[labels[index]]
        plt.xlabel(label, color=color)
    plt.show()


def run():
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot", ]
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    print(X_train.shape, y_train.shape)
    plot_multiple_images(images=X_test, labels=y_test, class_names=class_names)

    X_train_norm = X_train / X_train.max()
    X_test_norm = X_test / X_test.max()

    y_train_one_hot = tf.one_hot(y_train, depth=10)

    img_shape = (28, 28)
    tf.random.set_seed(42)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=img_shape),
        tf.keras.layers.Dense(40, activation="relu"),
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dense(len(class_names), activation="softmax"),
    ])
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    model.fit(X_train_norm, y_train_one_hot, epochs=10, workers=-1)


    print("multiclass classification")
