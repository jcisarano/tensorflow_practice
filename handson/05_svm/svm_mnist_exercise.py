# Exercise: Train an SVM classifier on the MNIST dataset.
# Since SVM classifiers are binary classifiers, you will need to use one-versus-all to classify all 10 digits.
# You may want to tune the hyperparameters using small validation sets to speed up the process. What accuracy can you reach?
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import load_digits, fetch_openml


def run():
    digits = load_digits(as_frame=False)
    digits = fetch_openml("mnist_784", version=1, cache=True, as_frame=False)
    # _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    # for ax, image, label in zip(axes, digits["data"], digits["target"]):
    #     ax.set_axis_off()
    #     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    #     ax.set_title("Training: %i" % label)
    # plt.show()

    X = digits["data"]
    y = digits["target"]



