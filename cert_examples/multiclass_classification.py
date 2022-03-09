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

    print("multiclass classification")
